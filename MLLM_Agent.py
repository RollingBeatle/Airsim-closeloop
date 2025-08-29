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

    def __init__(self, prompt, api_file, debug=False):
        # prompt, api key
        self.prompt = prompt
        self.api_file = api_file
        self.debug = debug
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
    Answer: str
    Indices: List[str]

ResponseFormatDecision = {
    "type": "json_schema",
    "json_schema": {
        "name": "landing_decision",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "minimum": 0, "description": "Chosen candidate index "},
                "reason": {"type": "string", "description": "Reason for choice"}
            },
            "required": ["index", "reason"],
            "additionalProperties": False
        }
    }
}

class GPTAgent(MLLMAgent):

    def get_mllm_agent(self):
        with open(self.api_file, "r") as f:
            api_key = f.read().strip()

        clientGPT = OpenAI(api_key=api_key)
        self.client = clientGPT
        # models = self.client.models.list()
        # for mod in models:
        #     print(mod)
        return clientGPT
    
    # Sends a call to the MLLM, 
    def mllm_call(self,detections, coversation_prompt, bb=None, full_img=None):
        # Get GPT Client
        clientGPT = self.get_mllm_agent()
        print("debug is active", self.debug)           
        # we want to send at most 5 areas to the LLM
        if len(detections) > 5:
            sorted_images_by_area = sorted(detections, key=lambda img: img.width * img.height, reverse=True)
            detections = sorted_images_by_area[:5]
        if self.debug:
            for det in range(len(detections)):
                print("The index is ", det)
                detections[det].show()
                input("Press enter to continue") 
        message =[
                    {"type": "image_url", "image_url": {"url":self.format_image(det)}}
                    for det in detections  
                ] 
        prompt_m =  [{"type": "text", "text": coversation_prompt}]
        message.extend(prompt_m)
        if bb:
            message + [{"type": "text", "text": bb}]
        if full_img: 
            message.extend([{"type": "image_url", "image_url": {"url":self.format_image(full_img)}}])        


        resp = self.completion_retry(
        content=message,
        model="gpt-5", clientGPT=clientGPT, # gpt-4.1-2025-04-14
        response_format=ResponseFormatBasic
        )

        result = json.loads(resp.choices[0].message.content)
        rich.print(result)  
        index = int(result['Indices'][0])
        if len(detections) == 1:
            index = 0
        try:
            if self.debug:
                print("The selected image")
                detections[index].show()
            print(int(result['Indices'][0]))
            return detections[index], int(result['Indices'][0]), result['Answer']
        # return result['Coordinates'][0]
        except:
            return None, 0, result['Answer']
    

    def log_errors(self, retry_state):
        print(
        "Request failed. Current retry attempts:{}. Sleep for {:.2f}. Exception: {}".format(
            retry_state.attempt_number, retry_state.idle_for, repr(retry_state.outcome.exception())
        )
    )
    
    
    def completion_retry(self, content, model, clientGPT:OpenAI, response_format=NOT_GIVEN):
        response = clientGPT.chat.completions.parse(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            
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
    
    



    
        
    