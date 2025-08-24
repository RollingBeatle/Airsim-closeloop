
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

from LiDAR.Get_data import get_image_lidar
from LiDAR.LLM_subimages import find_roofs
from drone_movement import DroneMovement
from MLLM_Agent import GPTAgent
from image_processing import ImageProcessing
from prompts import PROMPTS

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
    (-151.54669189453125, -43.83946990966797, -90.5884780883789),
    ( 6.851377487182617, -191.2527313232422, -105.62255096435547)
]
# Drone Camera Settings
# -----------------------------------------
IMAGE_WIDTH   = 1080
IMAGE_HEIGHT  = 1920
FOV_DEGREES   = 90
CAM_NAME      = "bottom_center"
EXAMPLE_POS   = (0,-35,-100)
# Directories Configuration
# -----------------------------------------
DELETE_LZ = True
DIRS = ["images", "landing_zones","point_cloud_data"]
# MLLM configurations
# -----------------------------------------
PROMPT_NAME = 'prompt1'
API_FILE = "my-k-api.txt"
# creates necessary directories
def create_subdirs():
    """Create subdirectories if they do not exist."""
    curr_dir = os.getcwd()
    for dir_name in DIRS:
        new_dir = os.path.join(curr_dir, dir_name)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(f"Created folder: {dir_name}")

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
def clear_dirs():
    """Clear existing data"""
    curr_dir = os.getcwd()
    for dir_name in DIRS:
        del_dir = os.path.join(curr_dir, dir_name)
        if os.path.exists(del_dir):
            shutil.rmtree(del_dir)
            print(f"Deleted: {del_dir}")
        else:
            print(f"Folder does not exist: {del_dir}")

   

def main_pipeline():

    # First load the prompt
    prompt = PROMPTS[PROMPT_NAME]

    # create necessary classes
    MLLM_Agent = GPTAgent(prompt, API_FILE, debug=DEBUG)
    processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=DEBUG)
    drone = DroneMovement()

    # set testing vars
    test = True
    iterations = 10 if test else 1

    # clear and create data
    if DELETE_LZ: clear_dirs()
    create_subdirs()

    # start pipeline
    for i in range(0,iterations):    
        # Position the drone
        if MOVE_FIXED:
            # drone.position_drone(fixed=False,position=POSITIONS[1])
            drone.take_off()
            time.sleep(3) # so the drone has time to stabalize
        elif MANUAL:
            drone.manual_control()

        if LIDAR:
            # LiDAR pipeline
            pc_name, img_name = "point_cloud_1", "img_1"
            get_image_lidar(pc_name,img_name)
            cv2_image = cv2.imread(os.path.join('images', f'{img_name}.png'))
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
            while abs(curr_height) > 10:
                # getting image from drone TODO: optimize image type
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]

                img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
                pillow_img = Image.fromarray(img)
                np_arr = np.array(pillow_img)
                # save image
                cv2.imwrite(os.path.join('images', 'mono.jpg'), cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                # surface crop
                bounding_boxes = None
                depth_map = None
                depth_raw = None
                if DEPTH_ONLY_PIPELINE or ALT_PIPELINE:
                    # depth map image and segmentation
                    pose = drone.client.getMultirotorState().kinematics_estimated.position
                    depth_raw, depth_map = processor.depth_analysis_depth_anything(image = pillow_img, max_depth=148) # depth_raw is metric depth_map is relative
                    img2 = np.array(depth_map)
                    orientation = drone.client.getMultirotorState().kinematics_estimated.orientation
                    surface_height = drone.get_rangefinder()
                    z_map, s = processor.get_Z_Values_From_Depth_Map_2(surface_height, depth_raw, pose.z_val)
                    z_map_vec = np.vectorize(z_map)
                    # apply to entire array
                    depth_mapped = z_map_vec(depth_raw)
                    # get boxes of surfaces
                    # areas = processor.segment_surfaces(img2, np_arr)
                    areas = processor.find_landing_zones(depth_mapped)
                    areas = processor.merge_landing_zones(areas)
                    # crop
                    img_copy = np_arr.copy()
                    
                    processor.crop_surfaces(areas, img_copy)           
                    # read saved detections
                    detections = [Image.fromarray(cv2.imread(os.path.join("./"+DIRS[1], f)))
                                for f in os.listdir("./"+DIRS[1])]
                    # act if the distance to the ground is a threshold or there are no detections
                    if not detections or curr_height < 20 :
                        detections, bounding_boxes = processor.crop_five_cuadrants(os.path.join('images', 'mono.jpg'))
                # only crop does not use depth map
                elif ONLY_CROP_PIPELINE:
                    detections, bounding_boxes = processor.crop_five_cuadrants(os.path.join('images', 'mono.jpg'))
                
                # ask LLM for surface
                select_pil_image, index, ans = MLLM_Agent.mllm_call(detections)   
                # get the pixels for the selected image
                if bounding_boxes:
                    print("the index of the image",index)
                    px = ((bounding_boxes[index][3] + bounding_boxes[index][1])//2) 
                    py = ((bounding_boxes[index][2] + bounding_boxes[index][0])//2) 
                    print("pixels",px,py)

                elif DEPTH_ONLY_PIPELINE:
                    px, py = processor.match_areas(areas,select_pil_image)
                # do the IPM to get the coordinates
                px,py = (520,520)
                pose = drone.client.getMultirotorState().kinematics_estimated.position
                orientation = drone.client.getMultirotorState().kinematics_estimated.orientation
                surface_height = drone.get_rangefinder()
                z_map, s = processor.get_Z_Values_From_Depth_Map_2(surface_height, depth_raw, pose.z_val)
                landing_zone_height = z_map(np.array(depth_raw)[px, py])
                if LANDING_ZONE_DEPTH_ESTIMATED:
                    tx, ty, tz = processor.inverse_perspective_mapping(pose, px, py, landing_zone_height, orientation)
                else: 
                    tx, ty, tz = processor.inverse_perspective_mapping(pose, px, py, curr_height, orientation)
                # start descent or move to the x,y in crop pipeline
                if ONLY_CROP_PIPELINE:
                    if index == 4:
                        curr_height = drone.move_to_z(pose.z_val)
                    else:
                        drone.client.moveToPositionAsync(tx, ty, pose.z_val, 3).join();time.sleep(5)
                           
                elif ALT_PIPELINE and curr_height >= 10:

                    drone.client.moveToPositionAsync(tx, ty, pose.z_val, 3).join();time.sleep(5)
                    # crop for z displacement
                    detections, bounding_boxes = processor.crop_five_cuadrants(os.path.join('images', 'mono.jpg'))
                    select_pil_image, index, ans = MLLM_Agent.mllm_call(detections) 
                    # checking if the selected space is the center
                    if index == 4:
                        curr_height = drone.move_to_z(pose.z_val)
                else:
                    # moving to desired z
                    curr_height = drone.move_drone(tx,ty,tz)
                # saving test results
                if test:
                    cv2.imwrite(os.path.join('tests', f"mono_depth{request_counter}_it_{i}.jpg"), cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                    select_pil_image.save(os.path.join('tests', f"mono_depth{request_counter}_it_{i}.jpg"))
                    start_data_rec("depth",i,request_counter,ans)
                request_counter+=1
                # clearing data    
                if DEBUG:
                    input("To continue and delete images, press enter")
                clear_dirs()
                create_subdirs()

            drone.land_drone()



        

if __name__ == "__main__":
    
    main_pipeline()
