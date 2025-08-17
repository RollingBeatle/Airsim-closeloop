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

# Define a simplified response format compatible with OpenAI structured outputs
ResponseFormatDecision = {
    "type": "json_schema",
    "json_schema": {
        "name": "landing_decision",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "reject": {"type": "boolean", "description": "True if no candidate is safe."},
                "index": {"type": "integer", "minimum": 0, "description": "Chosen candidate index if reject is false."},
                "reason": {"type": "string", "description": "Reason for choice if reject is false."},
                "reject_reason": {"type": "string", "description": "Reason for rejection if reject is true."}
            },
            "required": ["reject"]
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
    
    def mllm_call(self, prompt, candidates, annotated_image):
        """
        prompt: Dictionary or string containing the prompt/envelope
        candidates: List[Candidate] objects
        annotated_image: PIL.Image or np.ndarray (the annotated image with indices)
        """
        clientGPT = self.get_mllm_agent()
        # Convert annotated image to PIL if needed
        if isinstance(annotated_image, np.ndarray):
            annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        else:
            annotated_pil = annotated_image

        # Convert prompt to string if it's a dictionary
        if isinstance(prompt, dict):
            prompt_text = json.dumps(prompt, indent=2)
        else:
            prompt_text = str(prompt)

        resp = self.completion_retry(
            content=[
                {"type": "image_url", "image_url": {"url": self.format_image(annotated_pil)}},
                {"type": "text", "text": prompt_text}
            ],
            model="gpt-4o-2024-11-20", clientGPT=clientGPT,
            response_format=ResponseFormatDecision
        )

        result = json.loads(resp.choices[0].message.content)
        rich.print(result)
        
        # Handle rejection case
        if result.get("reject", False):
            return None, -1, result.get("reject_reason", "No safe landing zone found")
        
        # Handle success case - map index to candidate
        try:
            idx = int(result["index"])
            if idx < len(candidates):
                print("returning:",candidates[idx], idx, result.get("reason", ""))
                return candidates[idx], idx, result.get("reason", "")
            else:
                return None, -1, f"Invalid index {idx}, only {len(candidates)} candidates available"
        except (KeyError, ValueError, TypeError) as e:
            return None, -1, f"Error parsing response: {str(e)}"
    

    @staticmethod
    def log_errors(retry_state):
        print(
            "Request failed. Current retry attempts:{}. Sleep for {:.2f}. Exception: {}".format(
                retry_state.attempt_number, retry_state.idle_for, repr(retry_state.outcome.exception())
            )
        )
    
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        before_sleep=lambda retry_state: GPTAgent.log_errors(retry_state)
    )
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




    
        
    