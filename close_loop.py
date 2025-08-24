
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
import math
import open3d as o3d
from LiDAR.Get_data import get_image_lidar
from LiDAR.LLM_subimages import find_roofs
from drone_movement import DroneMovement
from MLLM_Agent import GPTAgent
from image_processing import ImageProcessing
from prompts_gt import PROMPTS, GROUND_TRUTH

from LiDAR.lidar_baseline import LidarMovement

# CONFIGURATION VARIABLES
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
MAX_HEIGHT = 155
# x, y, z, rotation: x,y,z,w
POSITIONS = [(-103.01084899902344, 20.440589904785156, -119.817626953125, 9.894594032999748e-10, 8.641491966443482e-09, 0.7200981974601746, 0.6938721537590027),
    #(-151.54669189453125, -43.83946990966797, -90.5884780883789),
    #( 6.851377487182617, -191.2527313232422, -105.62255096435547)
     (48.65536880493164, 80.24543762207031, -101.31468963623047,  -0.9999206066131592, -1.8329383522086573e-07, -2.4937296672078446e-08, 0.012602792121469975)
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
DIRS = ["images", "landing_zones","point_cloud_data", "tests"]
# MLLM configurations
# -----------------------------------------
PROMPT_NAME = 'prompt1'
API_FILE = "my-k-api.txt"
# Testing configurations
# ----------------------------------------
TESTING = True
INDIVIFUAL_MOD = True
CURRENT_SCENARIO = "scenario1"


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

# records experiment data for full pipeline
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

# Saves the data of the experiment to a csv file
def record_module_data(dirs, data):
    dirs = dirs+".csv"
    check_dir = os.path.isfile(dirs)
    print(f"Does the file for module {dirs} alredy exist:",check_dir)
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
def lidar_movement(client:airsim.MultirotorClient, px, py):
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

    lidar_m.crop_image_around_pixel(image, landing_center, size=100)

    # Get current GPS
    gps_data = client.getGpsData(gps_name="GPS", vehicle_name="airsimvehicle").gnss.geo_point
    current_gps = (gps_data.latitude, gps_data.longitude, gps_data.altitude)

    if landing_voxel is None:
        print("No valid landing voxel found.")
        return

    # Conversion: 1 meter ≈ 0.000009 deg lat, 0.000011 deg lon (approx.)
    METERS_TO_LAT = 0.000009
    METERS_TO_LON = 0.000011

    x_m, y_m, _ = landing_voxel  # x: right, y: forward in LIDAR

    offset_lat = y_m * METERS_TO_LAT
    offset_lon = -x_m * METERS_TO_LON

    target_lat = current_gps[0] + offset_lat
    target_lon = current_gps[1] + offset_lon

    # Estimate NED displacement from GPS difference (in meters)-y_m
    dy = -y_m
    #dx = x_m # because forward is negative NED x
    dx = -x_m
    print(f"Desired landing pixel: {landing_center}")
    print(f"Actual landing pixel: {landing_pixel}")
    print(f"Current GPS: {current_gps}")
    print(f"Target GPS: (lat: {target_lat}, lon: {target_lon})")
    print(f"Estimated move offset in NED frame: dx={dx:.2f}m, dy={dy:.2f}m")

    # Move using relative position (NED)
    current_pose = client.getMultirotorState().kinematics_estimated.position
    target_x = current_pose.x_val + dx
    target_y = current_pose.y_val + dy
    target_z = current_pose.z_val  # stay at same height
    return target_x, target_y, target_z

# Intersection over union score    
def iou(box1, box2):
    # box1 is ground truth
    x1_min, x1_max, y1_min, y1_max = box1
    y2_min, x2_min, y2_max, x2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymin = max(y1_min, y2_min)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0,inter_xmax - inter_xmin)
    inter_h = max(0,inter_ymax - inter_ymin)
    inter_a = inter_w * inter_h

    area_b1 = (x1_max-x1_min)*(y1_max-y1_min)
    area_b2 = (x2_max-x2_min)*(y2_max-y2_min)

    area_u =area_b1+area_b2 - inter_a
    iou = inter_a/area_u if area_u > 0 else 0.0
    return iou 

# detection module test
def detections_test(processor:ImageProcessing, drone:DroneMovement, it_numb, scenario="scenario1" ):
    pos = 0 if scenario=="scenario1" else 1
    ori = airsim.Quaternionr(
    x_val=POSITIONS[pos][3],
    y_val=POSITIONS[pos][4],
    z_val=POSITIONS[pos][5],
    w_val=POSITIONS[pos][6] ) if scenario =="scenario1" else None

    drone.position_drone(fixed=False,position=(POSITIONS[pos][0],POSITIONS[pos][1],POSITIONS[pos][2]), ori=ori)
    resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
    img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
    pillow_img = Image.fromarray(img)
    np_arr = np.array(pillow_img)
    # save image
    cv2.imwrite("tests/mono.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
    # surface crop
    bounding_boxes = None
    depth_map = processor.depth_analysis_depth_anything(pillow_img)
    img2 = np.array(depth_map)
    # get boxes of surfaces
    areas = processor.segment_surfaces(img2, np_arr)
    # crop
    img_copy = np_arr.copy()
    processor.crop_surfaces(areas, img_copy)           
    
    selected_area = []
    curr_max = 0.0
    box1 = (GROUND_TRUTH[scenario]['x_min'],GROUND_TRUTH[scenario]['x_max'],GROUND_TRUTH[scenario]['y_min'],GROUND_TRUTH[scenario]['y_max'])
    for ar in areas:
        score = iou(box1, ar)
        if score > curr_max:
            curr_max = score
            if len(selected_area) > 0:
                selected_area.pop()
            selected_area.append(ar)
    print(selected_area)            
    processor.crop_surfaces(selected_area,img_copy,f"test_{scenario}_{it_numb}")
    data = {
        "iteration":[it_numb],
        "iou_score": [curr_max],
        "crop_name": [f"test{it_numb}"],
        "landing_site": [scenario]
    }
    record_module_data("detections", data)



def llm_test(agent:GPTAgent, it_numb, scenario="scenario1" ):
    # cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    detections = [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(f"./samples/{scenario}", f)), cv2.COLOR_BGR2RGB))
    for f in os.listdir(f"./samples/{scenario}")]
    select_pil_image, index, ans = agent.mllm_call(detections, PROMPTS["conversation-1"]) 
    select_pil_image.save(f"tests/landing_zones/{scenario}_selected_{it_numb}.jpg")
    data = {
        "iteration":[it_numb],
        "selected_image": [f'{scenario}_selected_{it_numb}.jpg'],
        "correct": 0,
        "reason": [ans],
        "scenario": [scenario]
    }
    record_module_data('mllm_resp', data)

    
def landing_test(drone:DroneMovement, it_numb, scenario="scenario1"):

    pos = 0 if scenario=="scenario1" else 1
    ori = airsim.Quaternionr(
    x_val=POSITIONS[pos][3],
    y_val=POSITIONS[pos][4],
    z_val=POSITIONS[pos][5],
    w_val=POSITIONS[pos][6] ) if scenario =="scenario1" else None

    px, py = GROUND_TRUTH[scenario]['center_x'], GROUND_TRUTH[scenario]['center_y'] 
    drone.position_drone(fixed=False,position=(POSITIONS[pos][0],POSITIONS[pos][1],POSITIONS[pos][2]), ori=ori)
    tx, ty, tz = lidar_movement(drone.client, px,py)
    drone.client.moveToPositionAsync(tx, ty, tz, 3).join();time.sleep(5)
    drone.move_to_z(tz)
    current_pose = drone.client.getMultirotorState().kinematics_estimated.position
    actual_x, actual_y = current_pose.x_val, current_pose.y_val

    dist = math.sqrt((actual_x - GROUND_TRUTH[scenario]['x_real'])**2 + (actual_y - GROUND_TRUTH[scenario]['y_real'])**2)
    print("distance to x y ",dist)
    data = {
        "iteration":[it_numb],
        "actual_x": [actual_x],
        "actual_y": [actual_y],
        "distance_point": [dist],
        "scenario": [scenario]
    }
    record_module_data('landing', data)

def main_pipeline():

    # First load the prompt
    prompt = PROMPTS[PROMPT_NAME]

    # create necessary classes
    MLLM_Agent = GPTAgent(prompt, API_FILE, debug=DEBUG)
    processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=DEBUG)
    drone = DroneMovement()

    # set testing vars
    iterations = 20 if TESTING else 1

    if TESTING and INDIVIFUAL_MOD:
        for i in range(iterations):
            detections_test(processor, drone, i , CURRENT_SCENARIO)
        for i in range(iterations):
            llm_test(MLLM_Agent, i, scenario=CURRENT_SCENARIO)
        for i in range(iterations):
            landing_test(drone, i, scenario=CURRENT_SCENARIO)
        return
            

    # clear and create data
    if DELETE_LZ: clear_dirs()
    create_subdirs()

    # start pipeline
    for i in range(0,iterations):    
        # Position the drone
        if MOVE_FIXED:
            ori = None
            # if POSITIONS[0][3]:
            #     ori = airsim.Quaternionr(
            #     x_val=POSITIONS[1][3],
            #     y_val=POSITIONS[1][4],
            #     z_val=POSITIONS[1][5],
            #     w_val=POSITIONS[1][6])
            drone.position_drone(fixed=False,position=(POSITIONS[0][0],POSITIONS[0][1],POSITIONS[0][2]), ori=ori)
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
                depth_map = None
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
                    # [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join("./samples", f)), cv2.COLOR_BGR2RGB))
                    detections = [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join("./"+DIRS[1], f)),cv2.COLOR_BGR2RGB))
                                for f in os.listdir("./"+DIRS[1])]
                    # act if the distance to the ground is a threshold or there are no detections
                    if not detections or curr_height < 10 :
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
                    print("pixels",px,py)
                # do the IPM to get the coordinates
                pose = drone.client.getMultirotorState().kinematics_estimated.position
                surface_height = drone.get_rangefinder()
                z_map = processor.get_Z_Values_From_Depth_Map(abs(pose.z_val), surface_height, depth_map)
                landing_zone_height = z_map(np.array(depth_map)[py, px])
                if LANDING_ZONE_DEPTH_ESTIMATED:
                    tx, ty, tz = lidar_movement(drone.client, px,py)
                    #processor.inverse_perspective_mapping(pose, px, py, landing_zone_height)
                else: 
                    tx, ty, tz = processor.inverse_perspective_mapping(pose, px, py, curr_height)
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
                if TESTING and not INDIVIFUAL_MOD:
                    cv2.imwrite(f"tests/mono_depth{request_counter}_it_{i}.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                    select_pil_image.save(f"tests/selected_depth{request_counter}_it_{i}.jpg")
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
