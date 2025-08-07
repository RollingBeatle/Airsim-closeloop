"""
MLLM Agent for Autonomous Drone Recovery

Author: Diego Ortiz Barbosa
August 2025
"""
from abc import ABC, abstractmethod
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

class MLLMAgent(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_mllm_agent(self):
        pass

    @abstractmethod
    def mllm_call(self):
        pass
    
    @abstractmethod
    def format_image(self):
        pass

    @abstractmethod
    def log_errors(self):
        pass

class ResponseFormatBasic(BaseModel):
    Answer: List[str]
    Indices: List[str]


class GPTAgent(MLLMAgent):

    USE_MONOCULAR = True
    USE_STEREO = False
    LIDAR = False
    MOVE = True
    PROMPTS_FILE = 'prompts.json'

    def get_mllm_agent(self):

        with open("my-k-api.txt", "r") as f:
            api_key = f.read().strip()

        clientGPT = OpenAI(api_key=api_key)
        self.client = clientGPT
        return clientGPT
    
    def mllm_call(self, image, distance):
        # Get GPT Client
        clientGPT = self.get_mllm_agent()
        with open(self.PROMPTS_FILE, 'r') as f:
                # Parsing the JSON file into a Python dictionary
                prompts = json.load(f)

        if self.LIDAR:
            prompt = prompts["grid_prompt"]
        
        elif self.USE_MONOCULAR or self.USE_STEREO:
            prompt = prompts["basic_prompt"]
            
        landing_dir = './landing_zones'
        
        detections = [Image.fromarray(cv2.imread(os.path.join(landing_dir, f)))
                        for f in os.listdir(landing_dir)]
        if not detections or distance < 20 :
            # Crop the original image
            print("there are no detections, either detector failed or the whole surface is the detection\n We assume the second one")
            detection = Image.fromarray(cv2.imread(image))
            w, h = detection.size
            upper_left = detection.crop((0,0,w//2,h//2))
            print(f"The bounding box upper_left {upper_left}")
            upper_right = detection.crop((w//2,0,w,h//2))
            print(f"The bounding box upper_right {upper_right}")
            bottom_left = detection.crop((0,h//2,w//2,h))
            print(f"The bounding box bottom_left {bottom_left}")
            bottom_right = detection.crop((w//2,h//2,w,h))
            print(f"The bounding box bottom_left {bottom_right}")

            left = (w - 150) // 2
            top = (h - 150) // 2
            right = (w + 150) // 2
            bottom = (h + 150) // 2
            center = detection.crop((left,top,right,bottom))
            print(f"The bounding box center {center}")
            upper_left.show()
            upper_right.show()
            bottom_left.show()
            bottom_right.show()
            center.show()
            detections = [upper_left, upper_right, bottom_left, bottom_right, center]
        # we want to send at most 5 areas to the LLM
        if len(detections) > 5:
            sorted_images_by_area = sorted(detections, key=lambda img: img.width * img.height)
            detections = sorted_images_by_area[:4]
        
        
        resp = self.completion_retry(
        content=[
                    {"type": "image_url", "image_url": {"url": self.format_image(det)}}
                    for det in detections  
                ] + [{"type": "text", "text": prompt}],
        model="gpt-4o-2024-11-20", clientGPT=clientGPT,
        response_format=ResponseFormatBasic
        )

        result = json.loads(resp.choices[0].message.content)
        rich.print(result)  
        # detections[int(result['Indices'][0])].show()
        return detections[int(result['Indices'][0])]
        # return result['Coordinates'][0]
    
    def mllm_call_new(self,detections):
        # Get GPT Client
        clientGPT = self.get_mllm_agent()
        with open(self.PROMPTS_FILE, 'r') as f:
                # Parsing the JSON file into a Python dictionary
                prompts = json.load(f)

        if self.LIDAR:
            prompt = prompts["grid_prompt"]
        
        elif self.USE_MONOCULAR or self.USE_STEREO:
            prompt = prompts["basic_prompt"]
                    
        # we want to send at most 5 areas to the LLM
        if len(detections) > 5:
            sorted_images_by_area = sorted(detections, key=lambda img: img.width * img.height)
            detections = sorted_images_by_area[:4]
        
        
        resp = self.completion_retry(
        content=[
                    {"type": "image_url", "image_url": {"url": self.format_image(det)}}
                    for det in detections  
                ] + [{"type": "text", "text": prompt}],
        model="gpt-4o-2024-11-20", clientGPT=clientGPT,
        response_format=ResponseFormatBasic
        )

        result = json.loads(resp.choices[0].message.content)
        rich.print(result)  
        # detections[int(result['Indices'][0])].show()
        return detections[int(result['Indices'][0])]
        # return result['Coordinates'][0]
    

    def log_errors(self, retry_state):
        print(
        "Request failed. Current retry attempts:{}. Sleep for {:.2f}. Exception: {}".format(
            retry_state.attempt_number, retry_state.idle_for, repr(retry_state.outcome.exception())
        )
    )
    
    @retry(
    wait=wait_random_exponential(min=1, max=60),
    before_sleep=log_errors)
    def completion_retry(self, content, model, clientGPT, response_format=NOT_GIVEN):
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
    
    def format_image(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")

        image_bytes = buffered.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        url = f"data:image/jpeg;base64,{encoded_image}"
        return url




    
        
    