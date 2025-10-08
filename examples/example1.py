from LVLMRecovery import MLLM_Agent, image_processing, drone_movement, recovery_pipeline

IMAGE_WIDTH   = 960
IMAGE_HEIGHT  = 540
FOV_DEGREES   = 90
CAM_NAME      = "frontcamera"
position = (-103.01084899902344, 20.440589904785156, -119.817626953125)
system_prompt = """
# Safe Drone Landing Asistant
   You are a safety assistant for a quadcopter that is flying over a city and needs to perform an emergency landing on a surface. 
   The quadcopter will provide you with photos of several possible surfaces taken by its camera bellow, and ask questions of the suitability of the surfaces to perform the safest emergency landing possible.
   To classify as safe a potential landing site MUST take into consideration the following factors:

## Constraints           

    - **MUST** be clear of any obstructions such as air ducts, cars, rubble etc.
    - **MUST** be clear of people 
    - **MUST** be a flat surface

## Clarifications 
           
    - Ignore non-critical visual features such as shadows, surface textures, tile patterns, or color variations unless they **CLEARLY** indicate an actual obstruction.
    - Do not assume a surface is unsafe just because it might be something like a shingled roof unless there is strong visual evidence of danger (e.g., visible slope, fragile material, obvious gaps).
    - If no surface is perfectly safe, select the one with the lowest risk.
    - You **MUST** always return a clear selection (never refuse).
"""

def basic_pipeline():

    agent = MLLM_Agent.GPTAgent(system_prompt, "key.txt", debug=False)
    processor = image_processing.ImageProcessing(IMAGE_WIDTH,IMAGE_HEIGHT,FOV_DEGREES,debug=False)
    drone = drone_movement.DroneMovement()
    recovery_pipeline.main_pipeline("gpt-5",agent,processor,drone,position,None,1,1,False)

if __name__ == "__main__":
    basic_pipeline()