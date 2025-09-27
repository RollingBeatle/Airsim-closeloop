
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
ALT_PIPELINE = True
ONLY_CROP_PIPELINE = False

DEPTH_ONLY_PIPELINE = True
LIDAR = False
LANDING_ZONE_DEPTH_ESTIMATED = True
DEBUG = False
SEND_FULL = True
MARGINS = False
# Drone Movement Configurations
# -----------------------------------------
MOVE_FIXED = False
MANUAL = False      
MAX_HEIGHT = 155
# x, y, z, rotation: x,y,z,w
POSITIONS = [(-103.01084899902344, 20.440589904785156, -119.817626953125, 9.894594032999748e-10, 8.641491966443482e-09, 0.7200981974601746, 0.6938721537590027),
    #(-151.54669189453125, -43.83946990966797, -90.5884780883789),
    #( 6.851377487182617, -191.2527313232422, -105.62255096435547)
     (48.65536880493164, 80.24543762207031, -101.31468963623047,  0.00013935858441982418, -0.000704428821336478, -0.004044117871671915, 0.9999915957450867)
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
PROMPT_NAME = 'prompt1'
API_FILE = "my-k-api.txt"
# Testing configurations
# ----------------------------------------
TESTING = True
INDIVIFUAL_MOD = True
CURRENT_SCENARIO = "scenario1"
RANDOM_POS = False
HALTON_POS = True


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
def start_data_rec(dirs, it, rounds, just, ranks, resp_time):
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
def lidar_movement(client:airsim.MultirotorClient, processor:ImageProcessing, px, py):
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

def main_pipeline():
    # initial_movement, model, agent, processor, drone, crop_setting
    # First load the prompt
    prompt = PROMPTS[PROMPT_NAME]

    # create necessary classes
    MLLM_Agent = GPTAgent(prompt, API_FILE, debug=DEBUG)
    processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=DEBUG)
    drone = DroneMovement()

   
    iterations = 20 if TESTING else 1

    # clear and create data
    if DELETE_LZ: clear_dirs()
    create_subdirs() # 
    # if HALTON_POS:
    #     sites = load_halton_points()
    models = ['gpt-5', 'gpt-5-mini','gpt-5-nano']
    for m in models:

        # start pipeline CHECK ITERATION NUMBERS AND MODELS!!
        for i in range(0,iterations):    
            # Position the drone
            if MOVE_FIXED:
                ori = None
                pos = 0 if CURRENT_SCENARIO=="scenario1" else 1
                if pos == 0 or pos == 1:
                    ori = airsim.Quaternionr(
                    x_val=POSITIONS[pos][3],
                    y_val=POSITIONS[pos][4],
                    z_val=POSITIONS[pos][5],
                    w_val=POSITIONS[pos][6])
                drone.position_drone(fixed=False,position=(POSITIONS[pos][0],POSITIONS[pos][1],POSITIONS[pos][2]), ori=ori)
            elif MANUAL:
                drone.manual_control()
            elif RANDOM_POS:
                # -103.01084899902344, 20.440589904785156, -119.817626953125
                rand_x = -np.random.uniform(100, 300)
                rand_y = np.random.uniform(20, 100)
                position = (rand_x, rand_y, -130)
                print("The random position", position)
                drone.position_drone(fixed=False, position=position)
            elif HALTON_POS:
                halton_x = sites[i][0]
                halton_y = sites[i][1]
                position = (halton_x, halton_y, -130)
                print("The random position", position)
                drone.position_drone(fixed=False, position=position)
                

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
                llm_error = False
                print("Running the Combined pipeline")
                curr_height = drone.get_rangefinder()
                print("This is the height ", curr_height)
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
                    if DEPTH_ONLY_PIPELINE or ALT_PIPELINE:
                        # depth map image and segmentation
                        depth_map = processor.depth_analysis_depth_anything(pillow_img)
                        img2 = np.array(depth_map)
                        # get boxes of surfaces
                        areas = processor.segment_surfaces(img2, np_arr)
                        # crop
                        img_copy = np_arr.copy()
                        areas = processor.crop_surfaces(areas, img_copy  ) # , scale=1.3   REMEMBER TO CHECK THE SCALE   AND FULL IMAGE 
                        # read saved detections
                        # [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join("./samples", f)), cv2.COLOR_BGR2RGB))
                        detections = [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join("./"+DIRS[1], f)),cv2.COLOR_BGR2RGB))
                                    for f in os.listdir("./"+DIRS[1])]
                        # act if the distance to the ground is a threshold or there are no detections
                        if not detections or curr_height < 5 :
                            detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")                  
                    # ask LLM for surface
                    select_pil_image, index, ans, ranks, resp_time = MLLM_Agent.mllm_call(detections,  PROMPTS["conversation-1"], model=m )# full_img=np_arr, 
                    # get the pixels for the selected image if there are none from depth map
                    if bounding_boxes:
                        print("the index of the image",index)
                        try:
                            px = ((bounding_boxes[index][3] + bounding_boxes[index][1])//2) 
                            py = ((bounding_boxes[index][2] + bounding_boxes[index][0])//2) 
                        except:
                            print("selection falied, either a hallucination or llm decided that there is no suitable space")
                            start_data_rec(f"failed-iterations_{m}",i,request_counter,ans, ranks, resp_time)
                            llm_error = True
                            break
                        print("pixels",px,py)

                    elif DEPTH_ONLY_PIPELINE:
                        if select_pil_image == None:

                            print("selection falied, either a hallucination (or bad output) or llm decided that there is no suitable space")
                            start_data_rec(f"failed-iterations_{m}",i,request_counter,ans, ranks, resp_time)
                            llm_error = True
                            break

                        px, py = processor.match_areas(areas,select_pil_image)
                        print("pixels",px,py)
                    # do the IPM to get the coordinates
                    pose = drone.client.getMultirotorState().kinematics_estimated.position
                    
                    if LANDING_ZONE_DEPTH_ESTIMATED:
                        tx, ty, tz = lidar_movement(drone.client, processor, px, py)
                        #processor.inverse_perspective_mapping(pose, px, py, landing_zone_height)
                    else: 
                        tx, ty, tz = processor.inverse_perspective_mapping(pose, px, py, curr_height)
                    if TESTING and not INDIVIFUAL_MOD:
                        cv2.imwrite(f"tests/{m}/full_image_req{request_counter}_it_{i}.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                        select_pil_image.save(f"tests/{m}/selected_surface_req{request_counter}_it_{i}.jpg")
                        start_data_rec(f"pipeline_{m}",i,request_counter,ans, ranks, resp_time)
                    # start descent or move to the x,y in crop pipeline
                    if ONLY_CROP_PIPELINE:
                        if index == 4:
                            curr_height = drone.move_to_z(pose.z_val)
                        else:
                            drone.client.moveToPositionAsync(tx, ty, pose.z_val, 3).join();time.sleep(5)
                            
                    elif ALT_PIPELINE and curr_height >= 5:

                        drone.client.moveToPositionAsync(tx, ty, pose.z_val, 3).join();time.sleep(5)
                        # crop for z displacement
                        resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
                        img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
                        pillow_img = Image.fromarray(img)
                        np_arr = np.array(pillow_img)
                        # save image
                        cv2.imwrite("images/mono.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                        detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")
                        detections = [detections[4]]
                        select_pil_image, index, ans, ranks, resp_time = MLLM_Agent.mllm_call(detections,  PROMPTS["conversation-2"], model=m) 
                        # checking if the selected space is the center
                        if index == 1:
                            curr_height = drone.move_to_z(pose.z_val)
                    else:
                        # moving to desired z
                        curr_height = drone.move_drone(tx,ty,tz)
                    # saving test results
                    if TESTING and not INDIVIFUAL_MOD:
                        cv2.imwrite(f"tests/{m}/second_full_image_req{request_counter}_it_{i}.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
                        select_pil_image.save(f"tests/{m}/second_selected_closeup_req{request_counter}_it_{i}.jpg")
                        start_data_rec(f"pipeline_{m}",i,request_counter,ans, ranks, resp_time)
                    request_counter+=1
                    if request_counter == 10:
                            print("selection falied, either a hallucination or llm decided that there is no suitable space")
                            start_data_rec(f"request-out-iterations-{m}",i,request_counter,ans, ranks, resp_time)
                            llm_error = True
                            break
                    # clearing data    
                    if DEBUG:
                        input("To continue and delete images, press enter")
                    clear_dirs()
                    create_subdirs()
                if not llm_error:
                    drone.land_drone()
                else:
                    clear_dirs()
                    create_subdirs()
                print("ending iteration ", i)


if __name__ == "__main__":
    main_pipeline()
