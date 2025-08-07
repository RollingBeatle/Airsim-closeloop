#This combines both monocular and stereo landing pipelines for a drone using AirSim.
# It captures images, overlays a grid, asks an LLM for a target cell, and
# computes the drone's position to land accurately on the target cell.



import cosysairsim as airsim
import numpy as np
import cv2
import time
import math
import torch
from PIL import Image
from transformers import pipeline
from skimage import measure
import os
import shutil
# -------------------- CONFIG --------------------
IMAGE_WIDTH   = 960
IMAGE_HEIGHT  = 540
FOV_DEGREES   = 90
CAM_NAME      = "frontcamera"
BASELINE      = 2.0            # stereo baseline in meters
EXAMPLE_POS   = (0,-35,-100)
FIXED        = True        
DIRS = ["images", "landing_zones","point_cloud_data"]

# -------------------- HELPERS --------------------


def crop_surfaces(area, img):
    out = img.copy()
    i = 0
    for a in area:
        crop = out[a[0]:a[2], a[1]:a[3]]
        fname = f'landing_zones/landing_zone{i}.jpg'
        cv2.imwrite(fname, crop)
        i+=1

def create_subdirs():
    # add more dirs if needed
    curr_dir = os.getcwd()
    # create the dirs if not created yet
    for dir in DIRS:
        new_dir = curr_dir+f'/{dir}'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print(f"Created {dir} folder")

def clear_dirs():
    """Clear existing data"""
    curr_dir = os.getcwd()
    for dir in DIRS:
        del_dir = curr_dir+f'/{dir}'
        shutil.rmtree(del_dir)

# -------------------- DEPTH ESTIMATION -------------------

# Depth Anything V2
def depth_analysis_depth_anything(image:Image):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load pipeline
    print("Running pipeline....")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    # inference
    depth_image = pipe(image)["depth"]
    depth_image.save("images/depth_image.jpg")
    return depth_image

# segment images based on depth map
def segment_surfaces(img, original):
    
    depth = cv2.GaussianBlur(img, (5, 5), 0)

    # Compute gradient magnitude
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Threshold to get flat regions
    flat_mask = (grad_mag < 10).astype(np.uint8)
    flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # Label regions
    labeled = measure.label(flat_mask, connectivity=2)
    props = measure.regionprops(labeled)

    # Load original image for annotation
    annotated = original.copy()
    areas = []
    width_src, height_src= img.shape
    size = width_src*height_src
    # segment flat surfaces
    for p in props:
        if p.area > 700:  # filter out small noise
            minr, minc, maxr, maxc = p.bbox
            cv2.rectangle(annotated, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            if not size == maxc*maxr:
                areas.append((minr, minc, maxr, maxc))       
    # Save annotated image
    cv2.imwrite("images/flat_surfaces_annotated.jpg", annotated)
    return areas


def get_rangefinder(client:airsim.MultirotorClient):
    distance_sensor_data = client.getDistanceSensorData("Distance","airsimvehicle")
    distance = distance_sensor_data.distance
    print(f"Distance to ground: {distance:.2f}")
    return distance

    
# -------------------- MONOCULAR PIPELINE --------------------
def monocular_landing(llm_call, position):

    client = airsim.MultirotorClient()
    if position:
        position_drone(client)
    """ 
    We want to find a way to do ask the LLM x number of times before landing
    Rigth now we are proposing a z distance threshold to land:
    if surface ^ dist_z =< 10 => land()
    """
    # drone air-space limit
    distz = 155
    while abs(distz) > 20:
        
        # 1) capture and overlay
        resp = client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
        img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
        pillow_img = Image.fromarray(img)
        # depth map image and segmentation
        depth_map = depth_analysis_depth_anything(pillow_img)
        img2 = np.array(depth_map)
        # get boxes of surfaces
        np_arr = np.array(pillow_img)
        areas = segment_surfaces(img2, np_arr)
        # save image
        cv2.imwrite("images/mono.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
        # crop
        img_copy = np_arr.copy()
        crop_surfaces(areas, img_copy)
        # 2) ask LLM for surface
        select_pil_image = llm_call("images/mono.jpg",distz)      
        for area in areas:
            print(area)
            area_size = (area[3]-area[1])*(area[2]-area[0])
            img_size = select_pil_image.size[0]*select_pil_image.size[1] 
            if area_size == img_size:
                px = ((area[3] + area[1])//2) 
                py = ((area[2] + area[0])//2) 
                print(px,py)
                break  
        # 3) pixel→world XY via IPM
        pose = client.getMultirotorState().kinematics_estimated.position
        A = abs(pose.z_val)
        hFOV = math.radians(FOV_DEGREES)
        vFOV = 2*math.atan(math.tan(hFOV/2)*(IMAGE_HEIGHT/IMAGE_WIDTH))
        world_w = 2*A*math.tan(hFOV/2); world_h = 2*A*math.tan(vFOV/2)
        mpp_x = world_w/IMAGE_WIDTH; mpp_y = world_h/IMAGE_HEIGHT
        dx = px - IMAGE_WIDTH/2; dy = py - IMAGE_HEIGHT/2
        north = -dy*mpp_y; east = dx*mpp_x
        tx = pose.x_val + north; ty = pose.y_val + east; tz = pose.z_val

        # 4) move to the x,y postion
        client.moveToPositionAsync(tx,ty,tz,3).join(); time.sleep(1)
        
        # 5) go down to a desired z
        distz = get_rangefinder(client)
        print("distance to surface", distz)
        distz = tz + (get_rangefinder(client)*0.8) 
        print("target position", distz)
        client.moveToZAsync(distz,3).join()
        clear_dirs()
        create_subdirs()
    # client.moveToZAsync(distz,3).join()
    client.landAsync().join()
    

# -------------------- MOVEMENT ---------------------------
def position_drone(client:airsim.MultirotorClient):
    # Position the drone randomly in demo
    client.confirmConnection()
    client.enableApiControl(True); client.armDisarm(True)
    client.takeoffAsync().join(); time.sleep(1)
    if FIXED:
        x,y,z = EXAMPLE_POS
        client.moveToPositionAsync(x,y,z,3).join(); time.sleep(2)
    else:
        z0 = -np.random.uniform(40, 50)
        client.moveToZAsync(z0,2).join(); time.sleep(1)
    
def land_drone(client:airsim.MultirotorClient, x, y ,z): 

    client.moveToPositionAsync(x,y,z,3).join(); time.sleep(1)
    client.landAsync().join(); client.armDisarm(False)



