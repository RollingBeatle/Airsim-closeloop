
import os
import numpy as np
import cv2 
from PIL import Image
from openai import OpenAI, NOT_GIVEN
from tenacity import (
    retry,
    wait_random_exponential,
)
from io import BytesIO
from pydantic import BaseModel
from typing import List
import base64
import json
import rich
import pandas as pd

from Get_data import get_image_lidar
from LLM_subimages import find_roofs
import asyncio
# CONFIGURATION VARIABLES

USE_MONOCULAR = False
USE_STEREO = False
LIDAR = False

def gpt_call(clientGPT, image):

#     prompt = """A quadcopter is flying over a city and needs to perform an emergency landing on a rooftop.
# Given photos of several rooftops taken by this quadcopter, you are required to select the optimal rooftop as the emergency landing zone.

# Avoid any obstructions or people on the rooftop.

# First, explain your reasoning behind the rooftop ranking in detail, including the analysis for each rooftop. \
# Then, output the indices corresponding to each photo, representing the ranking from the most suitable rooftop to the least one.
# The index starts at 0.
# """
    prompt = """You have a photo of a drone camera view, and you need to assist a drone perform an emergency landing. Answer the following questions:
             Question 1: Describe each image
             Question 2: Determine which surface is more suitable to land.
             Question 3: Decide which surface to land on without hitting people or obstacles. Rank the index of the best suitable option as the first one and so on.
             Return an answer for each question."""
    if LIDAR:
        # with open(f"point_cloud_data/{pcd_file}", "rb") as f:
        #     pcd_upload = clientGPT.files.create(file=f,purpose="assistants")
        landing_dir = './landing_zones'
        detections = [Image.fromarray(cv2.imread(os.path.join(landing_dir, f)))
                      for f in os.listdir(landing_dir)]
        print(detections)
        resp = completion_retry(
        content=[
                    {"type": "image_url", "image_url": {"url": encode_image(det)}}
                    for det in detections  
                ] + [{"type": "text", "text": prompt}],
        model="gpt-4o-2024-11-20", clientGPT=clientGPT,
        response_format=ResponseFormat
        )
    
    result = json.loads(resp.choices[0].message.content)
    rich.print(result)  
        
def log_when_fail(retry_state):
    print(
        "Request failed. Current retry attempts:{}. Sleep for {:.2f}. Exception: {}".format(
            retry_state.attempt_number, retry_state.idle_for, repr(retry_state.outcome.exception())
        )
    )


@retry(
    wait=wait_random_exponential(min=1, max=60),
    before_sleep=log_when_fail
)
def completion_retry(content, model, clientGPT, response_format=NOT_GIVEN):
    response = clientGPT.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=2000,
        temperature=0,
        response_format=response_format
    )

    if response.choices[0].finish_reason != "stop":
        raise ValueError(
            "Generation finish reason: {}. {}".format(
                response.choices[0].finish_reason,
                response.usage.to_dict()
            )
        )

    return response

def encode_image(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")

    image_bytes = buffered.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")

    url = f"data:image/jpeg;base64,{encoded_image}"
    return url


# Specify the response format for GPT to make sure its output is structured as follows
class ResponseFormat(BaseModel):
    Answers: List[str]
class ResponseFormatBasic(BaseModel):
    Answer: str
    # Reason: str

def get_gpt_client():
    # read the API key and create the GPT client
    with open("my-k-api.txt", "r") as f:
        api_key = f.read().strip()

    clientGPT = OpenAI(api_key=api_key)
    return clientGPT

def main():
    # Get GPT Client
    client = get_gpt_client()
    # get data
    if LIDAR:
        # LiDAR pipeline
        pc_name, img_name = "point_cloud_1", "img_1"
        get_image_lidar(pc_name,img_name)
        cv2_image = cv2.imread(f'images/{img_name}.png')
        find_roofs(f"{pc_name}.pcd",f"{img_name}.png")
        image = Image.fromarray(cv2_image)
        result, justification = gpt_call(client, image)
        # show results
        rich.print(result, justification)

    elif USE_MONOCULAR:
        # Monocular pipeline
        pass
    elif USE_STEREO:
        # Stereo pipeline
        pass
    


    
  

if __name__ == "__main__":
    saved_dir = "C:/Users/Juan/Documents/modified_typefly/roof_attack/test_results/images"
    main()
