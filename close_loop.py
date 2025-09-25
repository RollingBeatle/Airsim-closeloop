
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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from openai import OpenAI
from LiDAR.Get_data import get_image_lidar
from LiDAR.LLM_subimages import find_roofs
from drone_movement import DroneMovement
from MLLM_Agent import GPTAgent
from image_processing import ImageProcessing
from prompts_gt import PROMPTS, GROUND_TRUTH

from LiDAR.lidar_baseline import LidarMovement
from reasons import reasons
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


def lidar_detections_test(processor:ImageProcessing, drone:DroneMovement, it_numb, scenario="scenario1" ):
    pos = 0 if scenario=="scenario1" else 1
    ori = airsim.Quaternionr(
    x_val=POSITIONS[pos][3],
    y_val=POSITIONS[pos][4],
    z_val=POSITIONS[pos][5],
    w_val=POSITIONS[pos][6] ) if scenario =="scenario1" else None

    drone.position_drone(fixed=False,position=(POSITIONS[pos][0],POSITIONS[pos][1],POSITIONS[pos][2]), ori=ori)
    pcd_name = f'pc_{it_numb}_{scenario}'
    img_name = f'img_{it_numb}_{scenario}'
    get_image_lidar(pcd_name, img_name, drone.client)
    areas = find_roofs(pcd_name+'.pcd',img_name+'.jpg')
    box1 = (GROUND_TRUTH[scenario]['x_min'],GROUND_TRUTH[scenario]['x_max'],GROUND_TRUTH[scenario]['y_min'],GROUND_TRUTH[scenario]['y_max'])
    selected_area = []
    for area in areas:
        score = iou(box1, area[0])
        if score > curr_max:
            curr_max = score
            if len(selected_area) > 0:
                selected_area.pop()
            selected_area.append(area)
    print(selected_area)            
    img_copy = cv2.imread(f"images/{img_name}.jpg",cv2.COLOR_BGR2RGB)
    processor.crop_surfaces(selected_area,img_copy,f"lidar_test_{scenario}_{it_numb}")
    data = {
        "iteration":[it_numb],
        "iou_score": [curr_max],
        "crop_name": [f"test{it_numb}"],
        "landing_site": [scenario]
    }
    record_module_data("detections_lidar", data)
        

def llm_test(agent:GPTAgent, it_numb, scenario="scenario1", expanded="" ):
    # cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) ,
    models = ['gpt-5', 'gpt-5-mini','gpt-5-nano']
    detections = [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(f"./samples/{scenario}", f)), cv2.COLOR_BGR2RGB))              
    for f in os.listdir(f"./samples/{scenario}")]
    for m in models:
        prompt = PROMPTS["conversation-1"]
        full_img = []
        if(SEND_FULL):
            full_img = cv2.imread(f"./samples/ground_truth_{scenario}.jpg", cv2.COLOR_BGR2RGB)
            prompt = PROMPTS["conversation-1-2"]
            expanded = "FI"
        if(MARGINS):
            detections = [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(f"./samples/{scenario}-{expanded}", f)), cv2.COLOR_BGR2RGB))              
                for f in os.listdir(f"./samples/{scenario}-{expanded}")]
            select_pil_image, index, ans, ranks, resp_time = agent.mllm_call(detections, prompt, full_img=full_img, model=m)
        else:
            select_pil_image, index, ans, ranks, resp_time = agent.mllm_call(detections, prompt, full_img=full_img, model=m) 
        
        select_pil_image.save(f"tests/landing_zones/{m}_{scenario}_selected_{it_numb}_{expanded}.jpg")
        data = {
            "iteration":[it_numb],
            "selected_image": [f'{m}_{scenario}_selected_{it_numb}_{expanded}.jpg'],
            "correct": 0,
            "reason": [ans],
            "response_time": [resp_time],
            "ranks": [ranks],
            "scenario": [scenario]
            
        }
        record_module_data(f'mllm_resp_{m}_{expanded}', data)

def llm_test_closeup(agent:GPTAgent, it_numb, processor,scenario="scenario1" ):

    models = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano']
    
    detections, _ = processor.crop_five_cuadrants(f"./samples/gt_closeup_{scenario}.jpg")
    detections = [detections[4]]
    for m in models:
        _, index, ans, ranks, response_time = agent.mllm_call(detections, PROMPTS["conversation-2"], model=m) 
        
        # select_pil_image.save(f"tests/landing_zones/close_up{scenario}_selected_{it_numb}.jpg")
        print("the index is", index)
        correct = 1 if index == 1 else 0
        data = {
            "iteration":[it_numb],
            "selected_image": [f'{scenario}_selected_{it_numb}.jpg'],
            "correct": correct,
            "reason": [ans],
            "response_time": [response_time],
            "ranks": [ranks],
            "scenario": [scenario]
            
        }
        record_module_data(f'mllm_resp_closeup_{m}', data)
    
def landing_test(drone:DroneMovement, it_numb, processor, scenario="scenario1"):

    pos = 0 if scenario=="scenario1" else 1
    ori = airsim.Quaternionr(
    x_val=POSITIONS[pos][3],
    y_val=POSITIONS[pos][4],
    z_val=POSITIONS[pos][5],
    w_val=POSITIONS[pos][6] ) if scenario =="scenario1" else None

    px, py = GROUND_TRUTH[scenario]['center_x'], GROUND_TRUTH[scenario]['center_y'] 
    drone.position_drone(fixed=False,position=(POSITIONS[pos][0],POSITIONS[pos][1],POSITIONS[pos][2]), ori=ori)
    tx, ty, tz = lidar_movement(drone.client, processor, px,py)
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

def load_halton_points():
    loaded_tuples = []
    with open("points_halton.txt", "r") as f:
        for line in f:
            x, y = line.strip().split(",")
            loaded_tuples.append((float(x), float(y)))
    return loaded_tuples

def main_pipeline():
    
    # First load the prompt
    prompt = PROMPTS[PROMPT_NAME]

    # create necessary classes
    MLLM_Agent = GPTAgent(prompt, API_FILE, debug=DEBUG)
    processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=DEBUG)
    drone = DroneMovement()

   
    iterations = 20 if TESTING else 1

    if TESTING and INDIVIFUAL_MOD:
        scenes = ["scenario1","scenario2"]
        for scene in scenes:
            # for i in range(iterations):
            #     detections_test(processor, drone, i , scenes)"30","40","50","60"
            for i in range(iterations):
                # margs = ["50"]
                # for marg in margs:
                    llm_test(MLLM_Agent, i, scenario=scene, expanded="")
            # for i in range(iterations):
            #     llm_test_closeup(MLLM_Agent, i, processor, scenario=scene)
            # for i in range(iterations):
            #     landing_test(drone, i, processor, scenario=scenes)
        return
            

    # clear and create data
    if DELETE_LZ: clear_dirs()
    create_subdirs() # 
    if HALTON_POS:
        sites = load_halton_points()
    models = ['gpt-5', 'gpt-5-mini','gpt-5-nano']
    for m in models:

        # start pipeline CHECK ITERATION NUMBERS AND MODELS!!
        for i in range(12,iterations):    
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
                    # only crop does not use depth map
                    elif ONLY_CROP_PIPELINE:
                        detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")
                    
                    # ask LLM for surface
                    select_pil_image, index, ans, ranks, resp_time = MLLM_Agent.mllm_call(detections,  PROMPTS["conversation-1"], model=m )# full_img=np_arr, 
                    # get the pixels for the selected image
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
                    surface_height = drone.get_rangefinder()
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

def embeddings():
    from reasons import final_des
    with open(API_FILE, "r") as f:
            api_key = f.read().strip()
    client = OpenAI(api_key=api_key)
    embeddings = []
    for text in final_des:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)
    plt.rcParams.update({
        "font.family": "serif",
        "text.usetex": False,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "grid.color": "gray",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.frameon": True,
    })
    embeddings = np.array(embeddings)
    # min_samples = 2
    # neighbors = NearestNeighbors(n_neighbors=min_samples, metric="cosine")
    # neighbors_fit = neighbors.fit(embeddings)
    # distances, indices = neighbors_fit.kneighbors(embeddings)

    from sklearn.decomposition import PCA

    pca = PCA().fit(embeddings)   # X = embeddings
    explained = np.cumsum(pca.explained_variance_ratio_)

    # plt.plot(np.arange(1, len(explained)+1), explained, marker='o')
    # plt.xlabel("Number of components")
    # plt.ylabel("Cumulative explained variance")
    # plt.grid()
    # plt.show()

    # pca with 13 dimensions then clustering in the final rankings (19 components)
    # variance gives 30 componentes -> 90% of variance, and we are sending 60
    pca = PCA(n_components=30)
    X_pca = pca.fit_transform(embeddings) 
    from sklearn.metrics import silhouette_score

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(X_pca, labels)
        print(f"k={k}, silhouette score={score:.3f}")
    
    kmeans2 = KMeans(n_clusters=2, random_state=42).fit(X_pca)
    kmeans3 = KMeans(n_clusters=3, random_state=42).fit(X_pca)
    labels2 = kmeans2.labels_
    labels3 = kmeans3.labels_

    # Reduce to 2D PCA for visualization only
    pca_vis = PCA(n_components=2, random_state=42)
    embeddings_2d = pca_vis.fit_transform(X_pca)

    # Plot K=2
    plt.figure(figsize=(12,5))

    # plt.subplot(1,2,1)
    # plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=labels2, cmap="tab10", s=80)
    # for i in range(len(embeddings_2d)):
    #     plt.annotate(f"P{i}", (embeddings_2d[i,0]+0.01, embeddings_2d[i,1]+0.01))
    # plt.title("KMeans Clusters (k=2)")

    # # Plot K=3
    # plt.subplot(1,2,2)
    custom_names = {
    2: "Cluster 2", #Single Candidate
    1: "Cluster 1", #Diverse Candidates
    0: "Cluster 0"  #Similar Candidates
}
    scatter = plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1], c=labels3, cmap="tab10", s=80)
    # for i in range(len(embeddings_2d)):
    #     plt.annotate(f"P{i}", (embeddings_2d[i,0]+0.01, embeddings_2d[i,1]+0.01))
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [custom_names[i] for i in range(3)], fontsize=14, loc='upper right')
    plt.xlabel("PC1", fontsize=18)
    plt.ylabel("PC2", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.title("KMeans Clusters (k=3)")

    plt.show()

    df = pd.DataFrame({
    "paragraph": final_des,
    "cluster_k2": labels2,
    "cluster_k3": labels3
    })

    # Save to CSV
    df.to_csv("confirmation_clusters.csv", index=False)
    # num_clusters = 2  # <-- choose the number of clusters you want
    # kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    # labels = kmeans.fit_predict(embeddings)

    # # 5. Print results
    # for para, label in zip(reasons, labels):
    #     print(f"Cluster {label}: {para[:100]}...")
    # for label in labels:
    #     print(label)
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=2, random_state=42)
    # reduced = pca.fit_transform(embeddings)

    # # Plot with cluster labels
    # plt.figure(figsize=(8, 6))
    # plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", s=80)

    # # Annotate each point with its paragraph index
    # for i, txt in enumerate(reasons):
    #     plt.annotate(f"P{i}", (reduced[i, 0]+0.01, reduced[i, 1]+0.01))

    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    # plt.title("PCA Visualization of Paragraph Clusters")
    # plt.show()

if __name__ == "__main__":
    embeddings()
    #main_pipeline()
