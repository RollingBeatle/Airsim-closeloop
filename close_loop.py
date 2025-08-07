
import os
import cv2 
from PIL import Image
import rich
import shutil

from LiDAR.Get_data import get_image_lidar
from LiDAR.LLM_subimages import find_roofs
from drone_movement import monocular_landing
from MLLM_Agent import GPTAgent

# CONFIGURATION VARIABLES
# TODO: integrate configuration into agent
USE_MONOCULAR = True
LIDAR = False
MOVE = True
PROMPTS_FILE = 'prompts.json'
DELETE_LZ = True
DIRS = ["images", "landing_zones","point_cloud_data"]



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

    
def main():
    MLLM_Agent = GPTAgent()
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

    elif USE_MONOCULAR:
        # Monocular pipeline
        monocular_landing(MLLM_Agent.mllm_call,MOVE)

def main_pipeline():

    MLLM_Agent = GPTAgent()
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

    elif USE_MONOCULAR:
        # Monocular pipeline
        monocular_landing(MLLM_Agent.mllm_call,MOVE)


if __name__ == "__main__":
    
    main()
