
import os
import cv2 
from PIL import Image
import rich
import shutil
import cosysairsim as airsim
import numpy as np

from LiDAR.Get_data import get_image_lidar
from LiDAR.LLM_subimages import find_roofs
from drone_movement import DroneMovement
from MLLM_Agent import GPTAgent
from image_processing import ImageProcessing

# CONFIGURATION VARIABLES
# TODO: integrate configuration into agent
PIPELINE = True
LIDAR = False
MOVE = True
PROMPTS_FILE = 'prompts.json'
DELETE_LZ = True
DIRS = ["images", "landing_zones","point_cloud_data"]
IMAGE_WIDTH   = 960
IMAGE_HEIGHT  = 540
FOV_DEGREES   = 90
CAM_NAME      = "frontcamera"
EXAMPLE_POS   = (0,-35,-100)
FIXED        = True        
MAX_HEIGHT = 155


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

    
def main_pipeline():

    MLLM_Agent = GPTAgent()
    processor = ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES)
    drone = DroneMovement()
    # get data
    if DELETE_LZ: clear_dirs()
    create_subdirs()
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

    elif PIPELINE:
        if MOVE:
            drone.position_drone()
        else:
            drone.position_drone(fixed=False)
        
        curr_height = drone.get_rangefinder()
        print("This is the height ", curr_height)
        while abs(curr_height) > 15:
            resp = drone.client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
            img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
            pillow_img = Image.fromarray(img)
            # depth map image and segmentation
            depth_map = processor.depth_analysis_depth_anything(pillow_img)
            img2 = np.array(depth_map)
            # get boxes of surfaces
            np_arr = np.array(pillow_img)
            areas = processor.segment_surfaces(img2, np_arr)
            # save image
            cv2.imwrite("images/mono.jpg", cv2.cvtColor(np_arr,cv2.COLOR_RGB2BGR))
            # crop
            img_copy = np_arr.copy()
            processor.crop_surfaces(areas, img_copy)
            # read saved detections
            detections = [Image.fromarray(cv2.imread(os.path.join("./"+DIRS[1], f)))
                        for f in os.listdir("./"+DIRS[1])]
            bounding_boxes = None
            if not detections or curr_height < 20 :
                detections, bounding_boxes = processor.crop_five_cuadrants("images/mono.jpg")
            # 2) ask LLM for surface
            
            select_pil_image, index = MLLM_Agent.mllm_call(detections)   
            if bounding_boxes:
                print("the index of the image",index)
                px = ((bounding_boxes[index][3] + bounding_boxes[index][1])//2) 
                py = ((bounding_boxes[index][2] + bounding_boxes[index][0])//2) 
                print("pixels",px,py)
            else:
                px, py = processor.match_areas(areas,select_pil_image)
            pose = drone.client.getMultirotorState().kinematics_estimated.position
            tx, ty, tz = processor.inverse_perspective_mapping(pose, px, py, curr_height)
            curr_height = drone.move_drone(tx,ty,tz)
            clear_dirs()
            create_subdirs()
        drone.land_drone()

        

if __name__ == "__main__":
    
    main_pipeline()
