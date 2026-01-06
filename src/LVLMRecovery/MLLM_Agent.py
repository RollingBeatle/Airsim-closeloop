"""
MLLM Agent for Autonomous Drone Recovery

Author: Diego Ortiz Barbosa
August 2025
"""
from abc import ABC, abstractmethod
import time
import numpy as np
import cv2 
from PIL import Image
from openai import OpenAI, NOT_GIVEN
from io import BytesIO
from pydantic import BaseModel
from typing import List
import base64
import json
import rich
import requests

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
        return clientGPT
    
    # Sends a call to the MLLM, 
    def mllm_call(self,detections, coversation_prompt, no_dets = False, full_img=[], model='gpt-5'):
        # Get GPT Client
        clientGPT = self.get_mllm_agent()
        print("debug is active", self.debug)    
             
        # we want to send at most 5 areas to the LLM
        if len(detections) > 5 and not no_dets:
            sorted_images_by_area = sorted(detections, key=lambda img: img.width * img.height, reverse=True)
            detections = sorted_images_by_area[:5]
        if self.debug:
            for det in range(len(detections)):
                print("The index is ", det)
                detections[det].show()
                input("Press enter to continue") 
        message = []
        for i in range(len(detections)):
            message.extend([{"type": "text", "text":f"[Image{i}]"},
                    {"type": "image_url", 
                     "image_url": {"url":self.format_image(detections[i])}
                     }])
            
        # message1 =[  {"type": "text", "text":"[Image]{}"},
        #             {"type": "image_url", "image_url": {"url":self.format_image(det)}}
        #             for det in detections  
        #         ] 
        prompt_m =  [{"type": "text", "text": coversation_prompt}]
        print("we are sending ", len(detections), ' surfaces')  
        print("current model: ", model)
        message.extend(prompt_m)
        # if bb:
        #     message + [{"type": "text", "text": bb}]
        if len(full_img)>0: 
            print("adding context")
            full_img = Image.fromarray(cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB))
            message.extend([{"type": "text", "text":f"[Context Image]"},
                            {"type": "image_url", "image_url": {"url":self.format_image(full_img)}}])        

        # print("message")
        # rich.print(message)
        start_time = time.time()

        resp = self.completion_retry(
        content=message,
        model=model, clientGPT=clientGPT, # gpt-4.1-2025-04-14
        response_format=ResponseFormatBasic
        )

        end_time = time.time()

        result = json.loads(resp.choices[0].message.content)
        rich.print(result)  
        index = int(result['Indices'][0])
        response_time = end_time - start_time
        if len(detections) == 1:
            index = 0
        try:
            if self.debug:
                print("The selected image")
                detections[index].show()
            print(int(result['Indices'][0]))
            return detections[index], int(result['Indices'][0]), result['Answer'], result['Indices'], response_time
        
        except Exception as e:
            print("Caught exception:", type(e).__name__, "-", e)
            return None, 0, result['Answer'], result['Indices'], response_time
    

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
    
  
class OpenRouterAgent(MLLMAgent):

    def get_mllm_agent(self):

        with open(self.api_file, "r") as f:
            api_key = f.read().strip()

        self.open_key = api_key    

    def mllm_call(self, detections, coversation_prompt, no_dets = False, full_img=[], model='gpt-5'):
        # get the connection to openrouter
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.open_key}",
            "Content-Type": "application/json",
            "Referer": "http://localhost",  
            "X-Title": "Test Script"
        }

        # we want to send at most 5 areas to the LLM
        if len(detections) > 5 and not no_dets:
            sorted_images_by_area = sorted(detections, key=lambda img: img.width * img.height, reverse=True)
            detections = sorted_images_by_area[:5]
        if self.debug:
            for det in range(len(detections)):
                print("The index is ", det)
                detections[det].show()
                input("Press enter to continue") 
        message = []
        for i in range(len(detections)):
            message.extend([
                # {"type": "text", "text":f"[Image{i}]"},
                    {"type": "image_url", 
                     "image_url": {"url":self.format_image(detections[i])}
                     }])
            
        prompt_m =  [{"type": "text", "text": coversation_prompt}]
        print("we are sending ", len(detections), ' surfaces')  
        print("current model: ", model)
        message.extend(prompt_m)
        
        if len(full_img)>0: 
            print("adding context")
            full_img = Image.fromarray(cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB))
            message.extend([
                {"type": "text", "text":f"[Context Image]"},
                            {"type": "image_url", "image_url": {"url":self.format_image(full_img)}}])
        # construct call 
        # rich.print(message)
        # print(message)       
        format_call_ans = self.format_call(message, model)
        # rich.print(format_call_ans)
        start_time = time.time()

        resp = requests.post(url=url,headers=headers,json=format_call_ans)
        
        data = resp.json()
        # # weather_info = data["choices"][0]["message"]["content"]
        print(data)
        # print(data['choices'][0]['message']['content'])
        # print("Answer", data['choices'][0]['message']['content']['Answer'])
        # print("Index", data['choices'][0]['message']['content']['Indices'])
        end_time = time.time()

        result = json.loads(data['choices'][0]['message']['content'])
        
        rich.print(result['Answer'])
        rich.print(result['Indices'])
        index = int(result['Indices'][0])
        response_time = end_time - start_time
        if len(detections) == 1:
            index = 0
        try:
            if self.debug:
                print("The selected image")
                detections[index].show()
            print(int(result['Indices'][0]))
            return detections[index], int(result['Indices'][0]), result['Answer'], result['Indices'], response_time
        
        except Exception as e:
            print("Caught exception:", type(e).__name__, "-", e)
            return None, 0, result['Answer'], result['Indices'], response_time
        
    def format_call(self, content, model):
        
        messages = [
                {
                    "role": "system",
                    "content": self.prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ] 
        request = {
            "model": model,
            "require_parameters": True,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "Rankings",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties":{
                            "Answer": {
                                "type": "string",
                                "description": "ranking reasons"
                            },
                            "Indices": {
                                "type": "array",
                                "items": {
                                    "type": "integer"
                                }
                            }
                        },
                        "required": ["Answer", "Indices"],
                        "additionalProperties": False,
                    }
                }
            }
        }  
        return request
    def log_errors(self, retry_state):
        print(
        "Request failed. Current retry attempts:{}. Sleep for {:.2f}. Exception: {}".format(
            retry_state.attempt_number, retry_state.idle_for, repr(retry_state.outcome.exception())
        )
    )
        
    def format_image(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")

        image_bytes = buffered.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        url = f"data:image/jpeg;base64,{encoded_image}"
        return url

class InterVLAgent(MLLMAgent):

    def get_mllm_agent(self):

        self.api_endpoint = "http://127.0.0.1:18000/infer"

    def mllm_call(self, detections, prompt, model, no_dets = False):
        self.get_mllm_agent()
        if len(detections) > 5 and not no_dets:
            sorted_images_by_area = sorted(detections, key=lambda img: img.width * img.height, reverse=True)
            detections = sorted_images_by_area[:2]
        if self.debug:
            for det in range(len(detections)):
                print("The index is ", det)
                detections[det].show()
                input("Press enter to continue") 
    
        images = [self.format_image(det) for det in detections]      
        
        payload = {
        "prompt": "Compare these images and describe differences.",
        "images_base64": images
        }
        start_time = time.time()
        r = requests.post(
            self.api_endpoint,
            json=payload,
            timeout=180
        )
        end_time = time.time()
        rich.print(r)
        print(r.json()["response"])
        print("--------")
        print(r.json()["indices"])
        print(r)
        index = int(r.json()["indices"][0])
        resp_time = end_time - start_time
        return detections[index], int(r.json()["indices"][0]), r.json()["response"], r.json()["indices"], resp_time
        

    def log_errors(self, retry_state):
        print(
        "Request failed. Current retry attempts:{}. Sleep for {:.2f}. Exception: {}".format(
            retry_state.attempt_number, retry_state.idle_for, repr(retry_state.outcome.exception())
        )
    )
    
        
    def format_image(self, pil_image):
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")

        image_bytes = buffered.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        url = f"data:image/jpeg;base64,{encoded_image}"
        return url



    
        
    