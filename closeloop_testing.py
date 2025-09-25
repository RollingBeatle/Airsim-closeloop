from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from openai import OpenAI
from prompts_gt import GROUND_TRUTH
from image_processing import ImageProcessing
import cv2
from PIL import Image
import os
import numpy as np
import pandas as pd
import cosysairsim as airsim
from drone_movement import DroneMovement
from LiDAR.Get_data import get_image_lidar
from LiDAR.LLM_subimages import find_roofs

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
    record_data("detections", data)

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

def main():
    crop_gt_surfaces(960,540,90, 1, scene=1)

if __name__ == "__main__":
    main()