
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

# CONFIGURATION VARIABLES
# TODO: integrate configuration into agent
# Pipeline Selection
# -----------------------------------------
COMBINED_PIPELINE = True
ALT_PIPELINE = False
ONLY_CROP_PIPELINE = True
DEPTH_ONLY_PIPELINE = False
LIDAR = False
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
IMAGE_WIDTH   = 960
IMAGE_HEIGHT  = 540
FOV_DEGREES   = 90
CAM_NAME      = "frontcamera"
EXAMPLE_POS   = (0,-35,-100)
# Directories Configuration
# -----------------------------------------
DELETE_LZ = True
DIRS = ["images", "landing_zones","point_cloud_data"]
# MLLM configurations
# -----------------------------------------
PROMPTS_FILE = 'prompts.json'
PROMPT_ONE = True
PROMPT_TWO = False
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
def clear_dirs():
    """Clear existing data"""
    curr_dir = os.getcwd()
    for dir in DIRS:
        del_dir = curr_dir+f'/{dir}'
        shutil.rmtree(del_dir)
   

def main_pipeline():

    # First load the prompt
    try:
        with open(PROMPTS_FILE, 'r') as f:
                # Parsing the JSON file into a Python dictionary
                prompts = json.load(f)
    except FileNotFoundError:
        print("prompt file not found")
        return
    if PROMPT_TWO:
            prompt = prompts["grid_prompt"]
        
    else:
            prompt = prompts["basic_prompt"]
    # create necessary classes
    MLLM_Agent = GPTAgent(prompt, API_FILE)
    processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=False)
    drone = DroneMovement()

    # set testing vars
    test = False
    iterations = 10 if test else 1

    # clear and create data
    if DELETE_LZ: clear_dirs()
    create_subdirs()

    # start pipeline
    for i in range(0,iterations):    
        # Position the drone
        if MOVE_FIXED:
            drone.position_drone(fixed=False,position=POSITIONS[1])
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
                # save image
                cv2.imwrite("images/mono.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                # surface crop
                bounding_boxes = None
                if DEPTH_ONLY_PIPELINE or ALT_PIPELINE:
                    # depth map image and segmentation
                    depth_map = processor.depth_analysis_depth_anything(pillow_img)
                    img2 = np.array(depth_map)
                    # get boxes of surfaces
                    areas = processor.segment_surfaces(img2, np_arr)
                    # crop
                    img_copy = np_arr.copy()
                    processor.crop_surfaces(areas, img_copy)           
                    # read saved detections
                    detections = [Image.fromarray(cv2.imread(os.path.join("./"+DIRS[1], f)))
                                for f in os.listdir("./"+DIRS[1])]
                    # act if the distance to the ground is a threshold or there are no detections
                    if not detections or curr_height < 20 :
                        detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")
                # only crop does not use depth map
                elif ONLY_CROP_PIPELINE:
                    detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")
                
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
                pose = drone.client.getMultirotorState().kinematics_estimated.position
                tx, ty, tz = processor.inverse_perspective_mapping(pose, px, py, curr_height)
                # start descent or move to the x,y in crop pipeline
                if ONLY_CROP_PIPELINE:
                    if index == 4:
                        curr_height = drone.move_to_z(pose.z_val)
                    else:
                        drone.client.moveToPositionAsync(tx, ty, pose.z_val, 3).join();time.sleep(5)
                           
                elif ALT_PIPELINE and curr_height >= 20:

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
                    cv2.imwrite(f"tests/mono_combined{request_counter}_it_{i}.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                    select_pil_image.save(f"tests/selected_combined{request_counter}_it_{i}.jpg")
                    start_data_rec("combined",i,request_counter,ans)
                request_counter+=1
                # clearing data    
                clear_dirs()
                create_subdirs()

            drone.land_drone()



        

if __name__ == "__main__":
    
    main_pipeline()
