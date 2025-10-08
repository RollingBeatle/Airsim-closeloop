
import os
import cv2 
from PIL import Image
import shutil
import cosysairsim as airsim
import numpy as np
import time
import pandas as pd
import os
import open3d as o3d
from typing import Tuple
from drone_movement import DroneMovement
from MLLM_Agent import GPTAgent
from image_processing import ImageProcessing
from util.prompts_gt import PROMPTS

from LiDAR.lidar_baseline import LidarMovement
# CONFIGURATION VARIABLES
# Pipeline Selection
# -----------------------------------------
DEBUG = False
# Drone Camera Settings
# -----------------------------------------
CAM_NAME = "frontcamera"
# Directories Configuration
# -----------------------------------------
DELETE_LZ = True
DIRS = ["images", "landing_zones","point_cloud_data"]

# creates necessary directories
def create_subdirs():
    """Create necessary directories"""
    # add more dirs if needed
    curr_dir = os.getcwd()
    # create the dirs if not created yet
    for dir in DIRS:
        new_dir = curr_dir+f'/{dir}'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(f"Created {dir} folder")

# records experiment data for full pipeline
def start_data_rec(dirs, it, rounds, just, ranks, resp_time):
    """record data into a csv file"""
    dirs = dirs+".csv"
    check_dir = os.path.isfile(dirs)
    print(check_dir)
    data = {
        "iteration":[it],
        "llm_rounds": [rounds],
        "justification": [just],
        "ranking": [ranks],
        "time": [resp_time],
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
   
# LiDAR based movement
def lidar_movement(client:airsim.MultirotorClient, processor:ImageProcessing, px, py):
    """IPM movement based on LiDAR"""
    # TODO: further optimize this, fix lidar color bug
    lidar_data = client.getLidarData('GPULidar1', 'airsimvehicle')
    lidar_m = LidarMovement()
    pcd_raw = lidar_m.get_open3d_point_cloud(lidar_data.point_cloud)
    pcd = lidar_m.rotate_point_cloud(pcd_raw)
    points = np.asarray(pcd.points)

    image = lidar_m.get_current_image(client, CAM_NAME)
    image = lidar_m.get_current_image(client, CAM_NAME)
    image = lidar_m.get_current_image(client, CAM_NAME)
    image = lidar_m.get_current_image(client, CAM_NAME)
    image = lidar_m.get_current_image(client, CAM_NAME)
    image = lidar_m.get_current_image(client, CAM_NAME)
    image = lidar_m.get_current_image(client, CAM_NAME)
    image = lidar_m.get_current_image(client, CAM_NAME)
    image = lidar_m.get_current_image(client, CAM_NAME)


    h_img, w_img = image.shape[:2]
    landing_center = (px,py)
    print(f"width: {w_img}, height: {h_img}")
    fx, fy = lidar_m.calculate_focal_length_from_fov(w_img, h_img, 90)
    pixel_to_coord = lidar_m.map_coord_to_pixel(image, points, fx, fy)
    landing_pixel, landing_voxel = lidar_m.find_closest_voxel_to_pixel(landing_center, pixel_to_coord)
    colored_pcd = lidar_m.colorize_point_cloud_with_image(points, image, fx, fy)
    if DEBUG:
        o3d.visualization.draw_geometries([colored_pcd])

    height_surface = landing_voxel[2]
    pose = client.getMultirotorState().kinematics_estimated.position
    orientation = client.getMultirotorState().kinematics_estimated.orientation
    tx, ty, tz = processor.inverse_perspective_mapping_v2(pose, landing_pixel[0], landing_pixel[1], height_surface, orientation)
    return tx, ty, tz

def main_pipeline(model:str, MLLM_Agent:GPTAgent, processor:ImageProcessing, drone:DroneMovement, position:Tuple[float, float, float],
                   orientation:airsim.Quaternionr, crop_setting:float, tracker:int, testing:bool):
    """
    Engages the LVLM assisted pipeline in the current location
    """
    # clear and create data
    if DELETE_LZ: clear_dirs()
    create_subdirs() # 
    # position drone
    drone.position_drone(fixed=False, position=position, ori=orientation)
    # start pipeline CHECK ITERATION NUMBERS AND MODELS!!
    llm_error = False
    print("Running the Pipeline")
    curr_height = drone.get_rangefinder()
    print("This is the distance to the closest surface ", curr_height)
    request_counter = 0
    while abs(curr_height) > 10:
        # getting image from drone TODO: optimize image type
        resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
        img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
        pillow_img = Image.fromarray(img)
        np_arr = np.array(pillow_img)
        # save image
        cv2.imwrite("images/mono.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
        # surface crop
        bounding_boxes = None
        depth_map = None
        # depth map image and segmentation
        depth_map = processor.depth_analysis_depth_anything(pillow_img)
        img2 = np.array(depth_map)
        # get boxes of surfaces
        areas = processor.segment_surfaces(img2, np_arr)
        # crop
        img_copy = np_arr.copy()
        areas = processor.crop_surfaces(areas, img_copy, scale=crop_setting) # , scale=1.3   REMEMBER TO CHECK THE SCALE   AND FULL IMAGE 
        # read saved detections
        detections = [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join("./"+DIRS[1], f)),cv2.COLOR_BGR2RGB))
                    for f in os.listdir("./"+DIRS[1])]
        # act if the distance to the ground is a threshold or there are no detections
        if not detections or curr_height < 5 :
            detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")                  
        # ask LLM for surface stage #1 ranking
        select_pil_image, index, ans, ranks, resp_time = MLLM_Agent.mllm_call(detections,  PROMPTS["conversation-1"], model=model )# full_img=np_arr, 
        # get the pixels for the selected image if there are none from depth map
        if bounding_boxes:
            # check if the bounding box actually exist and throw an error if they don't
            try:
                px = ((bounding_boxes[index][3] + bounding_boxes[index][1])//2) 
                py = ((bounding_boxes[index][2] + bounding_boxes[index][0])//2) 
            except:
                print("selection falied, either a hallucination or llm decided that there is no suitable space")
                start_data_rec(f"failed-iterations_{model}",tracker,request_counter,ans, ranks, resp_time)
                llm_error = True
                break
            print("pixels",px,py)
        # check if the selection output was an image, there should at least be one option
        if select_pil_image == None:
            # there is no suitable output so it means an error or hallucination, this is by design in this version
            print("selection falied, either a hallucination (or bad output) or llm decided that there is no suitable space")
            start_data_rec(f"failed-iterations_{model}",tracker,request_counter,ans, ranks, resp_time)
            llm_error = True
            break
        # match the selected image to its corresponding area
        px, py = processor.match_areas(areas,select_pil_image)
        print("destination pixels",px,py)
        # do the IPM to get the coordinates
        pose = drone.client.getMultirotorState().kinematics_estimated.position
        tx, ty, tz = lidar_movement(drone.client, processor, px, py)
        # check if testing mode on to record results
        if testing:
            cv2.imwrite(f"tests/{model}/full_image_req{request_counter}_it_{tracker}.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
            select_pil_image.save(f"tests/{model}/selected_surface_req{request_counter}_it_{tracker}.jpg")
            start_data_rec(f"pipeline_{model}",tracker,request_counter,ans, ranks, resp_time)
        # confirmation stage                
        if curr_height >= 5:
            # move to the desired x and y
            drone.client.moveToPositionAsync(tx, ty, pose.z_val, 3).join();time.sleep(5)
            # get the center subsquare of the image to confirm the suitability of the landing site
            resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
            img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
            pillow_img = Image.fromarray(img)
            np_arr = np.array(pillow_img)
            # save image
            cv2.imwrite("images/mono.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
            detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")
            # last image of the crop correspond to the center
            detections = [detections[4]]
            select_pil_image, index, ans, ranks, resp_time = MLLM_Agent.mllm_call(detections,  PROMPTS["conversation-2"], model=model) 
            # check that the response is successful to determine that the drone can start descend
            if index == 1:
                curr_height = drone.move_to_z(pose.z_val)
        else:
            # the drone is close enough to the surface to land
            curr_height = drone.move_drone(tx,ty,tz)
        # if testing mode active saving results
        if testing:
            cv2.imwrite(f"tests/{model}/second_full_image_req{request_counter}_it_{tracker}.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
            select_pil_image.save(f"tests/{model}/second_selected_closeup_req{request_counter}_it_{tracker}.jpg")
            start_data_rec(f"pipeline_{model}",tracker,request_counter,ans, ranks, resp_time)
        # adding to the request counter
        request_counter+=1
        # check if limit was reach and the selection failed
        if request_counter == 10:
                print("selection falied, either a hallucination or llm decided that there is no suitable space")
                start_data_rec(f"request-out-iterations-{model}",tracker,request_counter,ans, ranks, resp_time)
                llm_error = True
                break
        # clearing data    
        if DEBUG:
            input("To continue and delete images, press enter")
        clear_dirs()
        create_subdirs()
    # finally engage the land command
    if not llm_error:
        drone.land_drone()
    else:
        clear_dirs()
        create_subdirs()
        


if __name__ == "__main__":
    main_pipeline()
