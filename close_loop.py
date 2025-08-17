
import os
import cv2 
from PIL import Image
import rich
import shutil
import cosysairsim as airsim
import numpy as np
import time
import pandas as pd
import os
import json
import math

from LiDAR.Get_data import get_image_lidar
from LiDAR.LLM_subimages import find_roofs
from drone_movement import DroneMovement
from MLLM_Agent import GPTAgent
from image_processing import ImageProcessing
from prompts import PROMPTS, ENVELOPE, ResponseFormatDecision

# CONFIGURATION VARIABLES
# TODO: integrate configuration into agent
# Pipeline Selection
# -----------------------------------------
COMBINED_PIPELINE = True
ALT_PIPELINE = False
ONLY_CROP_PIPELINE = False
DEPTH_ONLY_PIPELINE = True
LIDAR = False
LANDING_ZONE_DEPTH_ESTIMATED = True
DEBUG = False
# Drone Movement Configurations
# -----------------------------------------
MOVE_FIXED = True
MANUAL = False
FIXED = True        
MAX_HEIGHT = 155
# x, y, z
POSITIONS = [
    (2498.781377, 606.094299, 2000),
    ( 6.851377487182617, -191.2527313232422, -105.62255096435547)
]
# Drone Camera Settings
# -----------------------------------------
IMAGE_WIDTH   = 960
IMAGE_HEIGHT  = 540
FOV_DEGREES   = 90
CAM_NAME      = "frontcamera"
EXAMPLE_POS   = (0,-35,-100)
# Directories Configuration
# -----------------------------------------
DELETE_LZ = False
DIRS = ["images", "landing_zones","point_cloud_data", "tests"]
# MLLM configurations
# -----------------------------------------
PROMPT_NAME = 'prompt1'
API_FILE = "my-k-api.txt"
# creates necessary directories
def create_subdirs():
    # add more dirs if needed
    curr_dir = os.getcwd()
    # create the dirs if not created yet
    for dir in DIRS:
        new_dir = curr_dir+f'/{dir}'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(f"Created {dir} folder")

# records experiment data
def start_data_rec(dirs, it, rounds, just):
    dirs = dirs+".csv"
    check_dir = os.path.isfile(dirs)
    print(check_dir)
    data = {
        "iteration":[it],
        "llm_rounds": [rounds],
        "justification": [just],
        "landing_site": 0
    }
    df = pd.DataFrame(data)
    if check_dir:
        df.to_csv(dirs, mode="a", header=False, index=False)
    else:
        df.to_csv(dirs, index=False)
# cleans workspace
def clear_dirs(preserve_images=True):
    """Clear existing data, optionally preserving images directory"""
    curr_dir = os.getcwd()
    for dir in DIRS:
        if preserve_images and dir == "images":
            continue  # Skip clearing images directory
        del_dir = curr_dir+f'/{dir}'
        if os.path.exists(del_dir):
            shutil.rmtree(del_dir)
   

def main_pipeline():

    # First load the prompt
    prompt = ENVELOPE

    # create necessary classes
    MLLM_Agent = GPTAgent(prompt, API_FILE, debug=DEBUG)
    processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=DEBUG)
    drone = DroneMovement()
    print("Capturing initial image...")
    init_pic = drone.client.simGetImages([
        airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene, False, False)
    ])[0]
    init_img = np.frombuffer(init_pic.image_data_uint8, np.uint8).reshape(
        init_pic.height, init_pic.width, 3
    )
    print("inital coordinates", drone.client.getMultirotorState().kinematics_estimated.position)
    init_img = cv2.cvtColor(init_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("images/first.jpg", init_img)

    # set testing vars
    test = True
    iterations = 3 if test else 1

    # clear and create data
    if DELETE_LZ: clear_dirs()
    create_subdirs()

    # start pipeline
    for i in range(0,iterations):    
        # Position the drone
        if MOVE_FIXED:
            print("Moving to fixed position")
            drone.position_drone(fixed=False,position=POSITIONS[0])
        elif MANUAL:
            drone.manual_control()

        if LIDAR:
            # LiDAR pipeline
            pc_name, img_name = "point_cloud_1", "img_1"
            get_image_lidar(pc_name,img_name)
            cv2_image = cv2.imread(f'images/{img_name}.png')
            find_roofs(f"{pc_name}.pcd",f"{img_name}.png")
            image = Image.fromarray(cv2_image)
            result, justification = MLLM_Agent.mllm_call(image)
            # show results
            rich.print(result, justification)
        
        # Main pipeline
        elif COMBINED_PIPELINE:
            print("Running the Combined pipeline")
            curr_height = drone.get_rangefinder()
            print("This is the height ", curr_height)
            request_counter = 0
            while abs(curr_height) > 15:
                # getting image from drone TODO: optimize image type
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
                pillow_img = Image.fromarray(img)
                np_arr = np.array(pillow_img)
                # save image with iteration info
                cv2.imwrite(f"images/mono_iter{i}_req{request_counter}.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                # surface crop
                bounding_boxes = None
                depth_map = None
                if DEPTH_ONLY_PIPELINE or ALT_PIPELINE:
                    # depth map image and segmentation
                    depth_map = processor.depth_analysis_depth_anything(pillow_img)
                    img2 = np.array(depth_map)
                    
                    # get boxes of surfaces
                    areas = processor.segment_surfaces(img2, np_arr)
                    
                    # crop
                   
                   
                   
                   ##Skipping cropping for now
                    # img_copy = np_arr.copy()
                    # processor.crop_surfaces(areas, img_copy)           
                    # # read saved detections
                    # detections = [Image.fromarray(cv2.imread(os.path.join("./"+DIRS[1], f)))
                    #             for f in os.listdir("./"+DIRS[1])]

                    #This now contains the candidates and correlating json to pass into LLM at once
                    detections = (areas)

                    # act if the distance to the ground is a threshold or there are no detections
                    if not detections or curr_height < 20 :
                        detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")
                # only crop does not use depth map
                elif ONLY_CROP_PIPELINE:
                    detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")
                
                # ask LLM for surface - try to find the latest annotated image
                annotated_image_path = f"images/flat_surfaces_annotated.jpg"
                if not os.path.exists(annotated_image_path):
                    # Fallback to any annotated image in the directory
                    import glob
                    annotated_files = glob.glob("images/flat_surfaces_annotated*.jpg")
                    if annotated_files:
                        annotated_image_path = annotated_files[-1]  # Use the latest
                annotated_image = cv2.imread(annotated_image_path)
                envelope = dict(ENVELOPE)
                envelope["input"] = dict(ENVELOPE["input"])
                envelope["input"]["candidates"]["count"] = len(detections)
                select_candidate, index, ans = MLLM_Agent.mllm_call(envelope, detections, annotated_image)
                
                # Create PIL image for saving in tests
                if annotated_image is not None:
                    select_pil_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                else:
                    # Fallback: use the mono depth image
                    select_pil_image = Image.fromarray(cv2.cvtColor(np_arr, cv2.COLOR_BGR2RGB))
                # get the pixels for the selected image
                if bounding_boxes:
                    print("the index of the image",index)
                    px = ((bounding_boxes[index][3] + bounding_boxes[index][1])//2) 
                    py = ((bounding_boxes[index][2] + bounding_boxes[index][0])//2) 
                    print("pixels",px,py)

                elif DEPTH_ONLY_PIPELINE:
                    if depth_map is None:
                        print("Error: depth_map is None!")
                        continue
                    
                    depth_array = np.array(depth_map)
                    if select_candidate is None:
                        print("No candidate selected, using center of image")
                        px, py = depth_array.shape[1] // 2, depth_array.shape[0] // 2
                    else:
                        # Note: center_xy gives (x, y) but depth_map is indexed as [y, x] or [row, col]
                        px, py = select_candidate.center_xy
                    print(f"Selected center coordinates: px={px}, py={py}")
                    print(f"Depth map shape: {depth_array.shape}")
                    
                    # Ensure coordinates are within bounds (swap px,py for numpy indexing)
                    if py >= depth_array.shape[0] or px >= depth_array.shape[1]:
                        print(f"Coordinates out of bounds! Using center of image instead.")
                        py = depth_array.shape[0] // 2
                        px = depth_array.shape[1] // 2
                    
                # do the IPM to get the coordinates
                pose = drone.client.getMultirotorState().kinematics_estimated.position
                
                surface_height = drone.get_rangefinder()
                print("Surface height", surface_height)
                print("Drone pose", pose)
                z_map = processor.get_Z_Values_From_Depth_Map(abs(pose.z_val), surface_height, depth_map)
                print("Z map", z_map)
               
                # Get depth value at the selected pixel (note: use py, px for row, col indexing)
                depth_value = depth_array[py, px]
                print(f"Depth value at ({py}, {px}): {depth_value}")
                landing_zone_height = z_map(depth_value)
                print(f"Calculated landing zone height: {landing_zone_height}")
                
                # Use camera altitude for IPM, not surface height
                camera_altitude = abs(pose.z_val)
                print(f"Camera altitude for IPM: {camera_altitude:.2f}m")
                
                if LANDING_ZONE_DEPTH_ESTIMATED:
                    tx, ty, tz = processor.inverse_perspective_mapping(pose, px, py, camera_altitude)
                    print(f"IPM with camera altitude: target=({tx:.2f}, {ty:.2f}, {tz:.2f})")
                    # Keep original z target from depth estimation
                    tz = pose.z_val  # Maintain current Z, move only in X,Y
                else: 
                    tx, ty, tz = processor.inverse_perspective_mapping(pose, px, py, camera_altitude)
                    print(f"IPM with camera altitude: target=({tx:.2f}, {ty:.2f}, {tz:.2f})")
                
                # Debug: Check coordinate transformations
                print(f"Pixel coordinates (x,y): ({px}, {py})")
                print(f"Image center: ({IMAGE_WIDTH//2}, {IMAGE_HEIGHT//2})")
                print(f"Offset from center: dx={px - IMAGE_WIDTH//2}, dy={py - IMAGE_HEIGHT//2}")
                print(f"Selected candidate bbox: {select_candidate.bbox_xyxy}")
                print(f"Final target position: ({tx:.2f}, {ty:.2f}, {tz:.2f})")
                
                # Calculate expected displacement
                world_w = 2 * camera_altitude * math.tan(math.radians(FOV_DEGREES/2))
                mpp_x = world_w / IMAGE_WIDTH
                expected_displacement = abs(px - IMAGE_WIDTH//2) * mpp_x
                print(f"Expected displacement: {expected_displacement:.2f}m (actual: {math.sqrt(tx**2 + ty**2):.2f}m)")
                # start descent or move to the x,y in crop pipeline
                if ONLY_CROP_PIPELINE:
                    if index == 4:
                        curr_height = drone.move_to_z(pose.z_val)
                    else:
                        drone.client.moveToPositionAsync(tx, ty, pose.z_val, 3).join();time.sleep(5)
                           
                elif ALT_PIPELINE and curr_height >= 10:

                    drone.client.moveToPositionAsync(tx, ty, pose.z_val, 3).join();time.sleep(5)
                    # crop for z displacement
                    detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")
                    select_pil_image, index, ans = MLLM_Agent.mllm_call(detections) 
                    # checking if the selected space is the center
                    if index == 4:
                        curr_height = drone.move_to_z(pose.z_val)
                else:
                    # moving to desired z
                    curr_height = drone.move_drone(tx,ty,tz)
                # saving test results
                if test:
                    cv2.imwrite(f"tests/mono_depth{request_counter}_it_{i}.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                    select_pil_image.save(f"tests/selected_depth{request_counter}_it_{i}.jpg")
                    start_data_rec("depth",i,request_counter,ans)
                request_counter+=1
                # clearing data    
                if DEBUG:
                    input("To continue and delete images, press enter")
                
            clear_dirs(preserve_images=True)  # Preserve images directory
            create_subdirs()
            drone.land_drone()



        

if __name__ == "__main__":
    
    main_pipeline()
