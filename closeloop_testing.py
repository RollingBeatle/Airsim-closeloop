from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from openai import OpenAI
from prompts_gt import GROUND_TRUTH, PROMPTS
from image_processing import ImageProcessing
import cv2
from PIL import Image
import os
import numpy as np
import pandas as pd
import cosysairsim as airsim
import math
import time
from drone_movement import DroneMovement
from MLLM_Agent import GPTAgent
from LiDAR.Get_data import get_image_lidar
from LiDAR.LLM_subimages import find_roofs
from LiDAR.lidar_baseline import LidarMovement
from close_loop import main_pipeline

## Camera settings
# -----------------------------------------
IMAGE_WIDTH   = 960
IMAGE_HEIGHT  = 540
FOV_DEGREES   = 90
CAM_NAME      = "frontcamera"
EXAMPLE_POS   = (0,-35,-100)

# Drone Movement Configurations
# -----------------------------------------
MOVE_FIXED = False
MANUAL = False      
MAX_HEIGHT = 155
# x, y, z, rotation: x,y,z,w
POSITIONS = [(-103.01084899902344, 20.440589904785156, -119.817626953125, 9.894594032999748e-10, 8.641491966443482e-09, 0.7200981974601746, 0.6938721537590027),
     (48.65536880493164, 80.24543762207031, -101.31468963623047,  0.00013935858441982418, -0.000704428821336478, -0.004044117871671915, 0.9999915957450867)
]
# Testing Configurations
# -----------------------------------------
SEND_FULL = False # Attach the full image to the request
MARGINS = False # Add additional context to the detected surface
DEBUG = False # 
# LVLM Configurations
# -----------------------------------------
PROMPT_NAME = 'prompt1'
API_FILE = "my-k-api.txt"


def crop_gt_surfaces(img_width, img_height, fov, scale, scene="1"):
    """Crop the ground truth surfaces with a determined padding scale"""
    processor = ImageProcessing(img_width, img_height, fov,debug=False)
    # load ground truth info
    full_img = cv2.imread(f"./samples/ground_truth_scenario1.jpg", cv2.COLOR_RGB2BGR)

    areas = [(GROUND_TRUTH[f'scenario{scene}']['y_min'], GROUND_TRUTH[f'scenario{scene}']['x_min'], GROUND_TRUTH[f'scenario{scene}']['y_max'], GROUND_TRUTH[f'scenario{scene}']['x_max']),
             (GROUND_TRUTH[f'scenario{scene}']['y_min_w'], GROUND_TRUTH[f'scenario{scene}']['x_min_w'], GROUND_TRUTH[f'scenario{scene}']['y_max_w'], GROUND_TRUTH[f'scenario{scene}']['x_max_w'])]
    
    # crop the images to the desired scale
    processor.crop_surfaces(areas, img=full_img, scale=scale) 


def tunning_par(processor:ImageProcessing ):
    """
    Applies the surface id part of the pipeline to the samples
    to help with internal tunning 
    """
    detections = [Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(f"./samples/general", f)), cv2.COLOR_BGR2RGB))
                  for f in os.listdir(f"./samples/general/")]
    pillow_img = detections[0]
    np_arr = np.array(pillow_img)
    depth_map = processor.depth_analysis_depth_anything(pillow_img)
    img2 = np.array(depth_map)
    # get boxes of surfaces
    areas = processor.segment_surfaces(img2, np_arr)
    # crop
    img_copy = np_arr.copy()
    processor.crop_surfaces(areas, img_copy) 

  
def iou(box1, box2):
    """
    Calculate the Jaccard index, box1 being the ground truth
    """
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

def detections_test(processor:ImageProcessing, drone:DroneMovement, it_numb, scenario="scenario1" ):
    """
    Test the detection module by taking a picture in airsim, can be changed by suppliying the ground truth
    """
    # resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
    img_sample = cv2.imread(f"./samples/ground_truth_{scenario}.jpg", cv2.COLOR_BGR2RGB)
    # img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
    pillow_img = Image.fromarray(img_sample)
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
    record_data("detections", data)

def lidar_detections_test(processor:ImageProcessing, drone:DroneMovement, it_numb, scenario="scenario1" ):
    """
    Test a possible LiDAR detection implementation
    """
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
    record_data("detections_lidar", data)

def record_data(dirs, data):
    """
    Save the results of a test into a csv file
    """
    dirs = dirs+".csv"
    check_dir = os.path.isfile(dirs)
    print(f"Does the file for module {dirs} alredy exist:",check_dir)
    df = pd.DataFrame(data)
    if check_dir:
        df.to_csv(dirs, mode="a", header=False, index=False)
    else:
        df.to_csv(dirs, index=False)

def llm_test(agent:GPTAgent, it_numb, scenario="scenario1", expanded="" ):
    """
    Test the individual LVLM module for the ranking stage
    """
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
        record_data(f'mllm_resp_{m}_{expanded}', data)

def llm_test_closeup(agent:GPTAgent, it_numb, processor,scenario="scenario1" ):
    """
    Test the individual LVLM module for the confirmation stage
    """
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
        record_data(f'mllm_resp_closeup_{m}', data)

def landing_test(drone:DroneMovement, it_numb, processor, scenario="scenario1"):
    """
    Test the conversion of the image space to the simulation space and its movement
    """
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
    record_data('landing', data)

def lidar_movement(client:airsim.MultirotorClient, processor:ImageProcessing, px, py):
    """
    LiDAR based movement single test
    """
    lidar_data = client.getLidarData('GPULidar1', 'airsimvehicle')
    lidar_m = LidarMovement()
    pcd_raw = lidar_m.get_open3d_point_cloud(lidar_data.point_cloud)
    pcd = lidar_m.rotate_point_cloud(pcd_raw)
    points = np.asarray(pcd.points)

    image = lidar_m.get_current_image(client, CAM_NAME)

    h_img, w_img = image.shape[:2]
    landing_center = (px,py)
    print(f"width: {w_img}, height: {h_img}")
    fx, fy = lidar_m.calculate_focal_length_from_fov(w_img, h_img, 90)
    pixel_to_coord = lidar_m.map_coord_to_pixel(image, points, fx, fy)
    landing_pixel, landing_voxel = lidar_m.find_closest_voxel_to_pixel(landing_center, pixel_to_coord)

    height_surface = landing_voxel[2]
    pose = client.getMultirotorState().kinematics_estimated.position
    orientation = client.getMultirotorState().kinematics_estimated.orientation
    tx, ty, tz = processor.inverse_perspective_mapping_v2(pose, landing_pixel[0], landing_pixel[1], height_surface, orientation)
    return tx, ty, tz

def load_halton_points():
    """
    Load Halton points for map covering testing
    """
    loaded_tuples = []
    with open("points_halton.txt", "r") as f:
        for line in f:
            x, y = line.strip().split(",")
            loaded_tuples.append((float(x), float(y)))
    return loaded_tuples

def modules_testing(iterations = 20, margs = ['']):
    """Test all the modules"""
    prompt = PROMPTS[PROMPT_NAME]

    MLLM_Agent = GPTAgent(prompt, API_FILE, debug=DEBUG)
    processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=DEBUG)
    drone = DroneMovement()
    scenes = ["scenario1","scenario2"]
    for scene in scenes:
        for i in range(iterations):
            detections_test(processor, drone, i , scenes)
        for i in range(iterations):
            for marg in margs:
                llm_test(MLLM_Agent, i, scenario=scene, expanded=marg)
        for i in range(iterations):
            llm_test_closeup(MLLM_Agent, i, processor, scenario=scene)
        for i in range(iterations):
            landing_test(drone, i, processor, scenario=scenes)    

def embeddings():
    """Get information from embeddings for clustering using the OpenAI API"""
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


def test_full_pipeline(iteration, mode="scenario1", save = False):
    """Full pipeline testing on predetermined scenarios and halton points"""
    
    if mode.lower() == 'scenario1':
        position = (POSITIONS[0][0],POSITIONS[0][1],POSITIONS[0][2])
        orientation = airsim.Quaternionr(
                x_val=POSITIONS[0][3],
                y_val=POSITIONS[0][4],
                z_val=POSITIONS[0][5],
                w_val=POSITIONS[0][6])
    elif mode.lower() == 'scenario2':
        position = (POSITIONS[0][0],POSITIONS[0][1],POSITIONS[0][2])
        orientation = airsim.Quaternionr(
                x_val=POSITIONS[1][3],
                y_val=POSITIONS[1][4],
                z_val=POSITIONS[1][5],
                w_val=POSITIONS[1][6])    
    elif mode.lower() == 'halton':
        sites = load_halton_points()
        halton_x = sites[iteration][0]
        halton_y = sites[iteration][1]
        position = (halton_x, halton_y, -130)
        orientation = None
    else:
        return Exception("Not a valid option")
    # create necessary classes
    prompt = PROMPTS[PROMPT_NAME]
    MLLM_Agent = GPTAgent(prompt, API_FILE, debug=DEBUG)
    processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=DEBUG)
    drone = DroneMovement()
    models = ['gpt-5', 'gpt-5-mini','gpt-5-nano']
    
    for model in models:
            main_pipeline(model,MLLM_Agent,processor,drone,position,orientation,1,iteration,save)

    
def main():
    # processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=DEBUG)
    # detections_test(processor,None,1)
    # crop_gt_surfaces(960,540,90, 1, scene=1)
    iterations = 20
    for i in range(iterations):
        test_full_pipeline(i,"scenario1")

if __name__ == "__main__":
    main()