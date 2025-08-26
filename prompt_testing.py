"""
Prompt Testing Framework for MLLM Agent
Simple framework to test different prompting strategies and record results to CSV

Author: Testing Framework
January 2025
"""

import os
import cv2
import pandas as pd
import json
import time
from PIL import Image
from MLLM_Agent import GPTAgent
from prompts_gt import PROMPTS, ENVELOPE
import numpy as np

# Configuration
API_KEY_FILE = "my-k-api.txt"
MODEL_NAME = "gpt-4o-2024-11-20"
NUM_ITERATIONS = 10 

# Different models to test
MODELS_TO_TEST = [
    "gpt-4o-2024-11-20",      # Current model
    "gpt-4o",                 # Standard GPT-4o
    "gpt-4o-mini",            # Faster, cheaper version
    "gpt-4-turbo",            # GPT-4 Turbo
]

def record_module_data(filename, data):
    """Saves the data of the experiment to a csv file"""
    filepath = f"{filename}.csv"
    check_dir = os.path.isfile(filepath)
    print(f"Does the file for module {filepath} already exist: {check_dir}")
    df = pd.DataFrame(data)
    if check_dir:
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)

def load_images_from_path(image_path):
    """Load images from path, handle both single images and directories"""
    images = []
    
    if os.path.isfile(image_path):
        # Single image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(Image.open(image_path))
        else:
            raise ValueError(f"Unsupported image format: {image_path}")
    elif os.path.isdir(image_path):
        # Directory of images
        for filename in sorted(os.listdir(image_path)):  # Sort for consistent ordering
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(image_path, filename)
                images.append(Image.open(img_path))
        if not images:
            raise ValueError(f"No image files found in directory: {image_path}")
    else:
        raise ValueError(f"Path does not exist: {image_path}")
    
    return images

def create_single_annotated_image(images):
    """
    Create a single annotated image with numbered bounding boxes from multiple images
    This simulates the single annotated strategy by combining multiple images into one
    """
    if not images:
        raise ValueError("No images provided")
    
    # For simplicity, we'll use the first image as base and add text annotations
    base_image = images[0].copy()
    
    # Convert PIL to OpenCV for drawing
    img_cv = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
    height, width = img_cv.shape[:2]
    
    # Create simple grid of bounding boxes for demonstration
    box_width = width // 3
    box_height = height // 3
    
    for i in range(min(4, len(images))):  # Max 4 boxes
        row = i // 2
        col = i % 2
        x1 = col * box_width + box_width // 4
        y1 = row * box_height + box_height // 4
        x2 = x1 + box_width // 2
        y2 = y1 + box_height // 2
        
        # Draw rectangle and number
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img_cv, str(i), (x1 + 10, y1 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Convert back to PIL
    annotated_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return annotated_pil

def test_multiple_images_strategy(images_path, prompt_key="prompt1", scenario="default", model="gpt-4o-2024-11-20"):
    """
    Strategy A: Multiple separate images for ranking
    Uses the current MLLM_Agent.py approach
    """
    print(f"\n=== Testing Multiple Images Strategy ===")
    print(f"Images path: {images_path}")
    print(f"Prompt: {prompt_key}")
    print(f"Model: {model}")
    
    # Initialize agent with specific model
    agent = GPTAgent(PROMPTS[prompt_key], API_KEY_FILE, debug=False, model=model)
    
    # Load images
    images = load_images_from_path(images_path)
    print(f"Loaded {len(images)} images")
    
    results = []
    
    for iteration in range(NUM_ITERATIONS):
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
        start_time = time.time()
        
        try:
            # Make the MLLM call with multiple images
            prompt_to_use = PROMPTS[prompt_key]
            # Convert dictionary prompts to JSON strings
            if isinstance(prompt_to_use, dict):
                prompt_to_use = json.dumps(prompt_to_use, indent=2)
                
            selected_image, selected_index, answer = agent.mllm_call(
                detections=images,
                coversation_prompt=prompt_to_use
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Record the result
            data = {
                "iteration": [iteration],
                "strategy": ["multiple_images"],
                "prompt_key": [prompt_key],
                "model": [model],
                "scenario": [scenario],
                "selected_index": [selected_index],
                "reason": [str(answer)],
                "response_time_seconds": [response_time],
                "success": [selected_image is not None],
                "image_path": [images_path],
                "num_images": [len(images)]
            }
            
            results.append(data)
            record_module_data(f'promptest_multi_{scenario}_{prompt_key}', data)
            
            print(f"  Selected index: {selected_index}, Time: {response_time:.2f}s")
            
        except Exception as e:
            print(f"  Error in iteration {iteration}: {str(e)}")
            data = {
                "iteration": [iteration],
                "strategy": ["multiple_images"],
                "prompt_key": [prompt_key],
                "model": [model],
                "scenario": [scenario],
                "selected_index": [-1],
                "reason": [f"Error: {str(e)}"],
                "response_time_seconds": [0],
                "success": [False],
                "image_path": [images_path],
                "num_images": [len(images)]
            }
            results.append(data)
            record_module_data(f'promptest_multi_{scenario}_{prompt_key}', data)
    
    return results

def test_single_annotated_strategy(image_path, prompt_key="prompt1", scenario="default", model="gpt-4o-2024-11-20"):
    """
    Strategy B: Single annotated image with numbered bounding boxes
    For this we'll pass a single image as a list to the mllm_call method
    """
    print(f"\n=== Testing Single Annotated Image Strategy ===")
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt_key}")
    print(f"Model: {model}")
    
    # Initialize agent with specific model
    agent = GPTAgent(PROMPTS[prompt_key], API_KEY_FILE, debug=False, model=model)
    
    # Load the annotated image
    if os.path.isfile(image_path):
        annotated_image = Image.open(image_path)
    else:
        # If it's a directory, create an annotated image from the first few images
        images = load_images_from_path(image_path)
        annotated_image = create_single_annotated_image(images)
    
    results = []
    
    for iteration in range(NUM_ITERATIONS):
        print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}")
        start_time = time.time()
        
        try:
            # Make the MLLM call with single annotated image (passed as a list)
            prompt_to_use = PROMPTS[prompt_key]
            # Convert dictionary prompts to JSON strings
            if isinstance(prompt_to_use, dict):
                prompt_to_use = json.dumps(prompt_to_use, indent=2)
                
            selected_image, selected_index, answer = agent.mllm_call(
                detections=[annotated_image],  # Pass as single-item list
                coversation_prompt=prompt_to_use
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Record the result
            data = {
                "iteration": [iteration],
                "strategy": ["single_annotated"],
                "prompt_key": [prompt_key],
                "model": [model],
                "scenario": [scenario],
                "selected_index": [selected_index],
                "reason": [str(answer)],
                "response_time_seconds": [response_time],
                "success": [selected_image is not None],
                "image_path": [image_path]
            }
            
            results.append(data)
            record_module_data(f'promptest_single_{scenario}_{prompt_key}', data)
            
            print(f"  Selected index: {selected_index}, Time: {response_time:.2f}s")
            
        except Exception as e:
            print(f"  Error in iteration {iteration}: {str(e)}")
            data = {
                "iteration": [iteration],
                "strategy": ["single_annotated"],
                "prompt_key": [prompt_key],
                "model": [model],
                "scenario": [scenario],
                "selected_index": [-1],
                "reason": [f"Error: {str(e)}"],
                "response_time_seconds": [0],
                "success": [False],
                "image_path": [image_path]
            }
            results.append(data)
            record_module_data(f'promptest_single_{scenario}_{prompt_key}', data)
    
    return results

def run_prompt_test(image_path, scenario="test_scenario", strategies=["multi"], prompts=["prompt1"], models=["gpt-4o-2024-11-20"]):
    """
    Main function to run prompt testing
    
    Args:
        image_path: Path to image file or directory of images
        scenario: Name for this test scenario
        strategies: List of strategies to test ["single", "multi"] 
        prompts: List of prompt keys to test from PROMPTS dictionary
        models: List of models to test
    """
    print(f"Starting prompt testing for scenario: {scenario}")
    print(f"Image path: {image_path}")
    print(f"Strategies: {strategies}")
    print(f"Prompts: {prompts}")
    print(f"Models: {models}")
    print(f"Iterations per test: {NUM_ITERATIONS}")
    
    all_results = []
    
    for model in models:
        for prompt_key in prompts:
            if prompt_key not in PROMPTS:
                print(f"Warning: Prompt '{prompt_key}' not found in PROMPTS dictionary")
                continue
                
            for strategy in strategies:
                if strategy == "single":
                    # Single annotated image strategy
                    results = test_single_annotated_strategy(image_path, prompt_key, scenario, model)
                    all_results.extend(results)
                        
                elif strategy == "multi":
                    # Multiple images strategy
                    results = test_multiple_images_strategy(image_path, prompt_key, scenario, model)
                    all_results.extend(results)
                    
                else:
                    print(f"Unknown strategy: {strategy}")
    
    print(f"\nTesting completed! Total tests run: {len(all_results)}")
    return all_results

if __name__ == "__main__":
    # Example usage - you can modify these paths and parameters
    
    # Test with multiple images from landing_zones directory
    # run_prompt_test(
    #     image_path="landing_zones/",  # directory with multiple cropped images
    #     scenario="multi_crops_test",
    #     strategies=["multi"], 
    #     prompts=["prompt1", "original"]
    # )
    
    # Test with a single annotated image
    # run_prompt_test(
    #     image_path="images/flat_surfaces_annotated.jpg",
    #     scenario="single_annotated_test",
    #     strategies=["single"],
    #     prompts=["prompt1"]
    # )
    
    # Test both strategies with the same image set
    # run_prompt_test(
    #     image_path="images/flat_surfaces_annotated.jpg",
    #     scenario="comparison_test",
    #     strategies=["single", "multi"],  # Test both strategies
    #     prompts=["prompt1"]
    # )
    
    # Test multiple cropped images strategy
    # run_prompt_test(
    #     image_path="landing_zones/",
    #     scenario="cropped_landing_zones",
    #     strategies=["multi"],
    #     prompts=["prompt1"]
    # )
    
    # Comprehensive Model Comparison Test
    # First add ENVELOPE to our prompt options as a string
    envelope_prompt = f"""
Task: {ENVELOPE['task']}

Input: {ENVELOPE['input']['description']}

Format: {ENVELOPE['format']['description']}
Expected response should include: index (integer), reason (string), and optionally ranking (array of integers).

Constraints:
{chr(10).join(f"- {constraint}" for constraint in ENVELOPE['constraints'])}

Example:
Input: {ENVELOPE['examples'][0]['input']}
Output: {json.dumps(ENVELOPE['examples'][0]['output'])}
"""
    
    PROMPTS["envelope"] = envelope_prompt
    
    # Test models to compare
    test_models = [
        "gpt-4o-2024-11-20",  # Current model
        "gpt-4o-mini",        # Faster, cheaper version
    ]
    
    print("🚁 === COMPREHENSIVE PROMPT TESTING WITH MULTIPLE MODELS ===")
    print(f"Testing {len(test_models)} models with multiple strategies and prompts")
    
    # Test 1: Single Annotated Strategy with both conversation and JSON prompts
    run_prompt_test(
        image_path="images/flat_surfaces_annotated.jpg",
        scenario="single_annotated_comparison",
        strategies=["single"],
        prompts=["prompt1", "envelope"],
        models=test_models
    )
    
    # Test 2: Multiple Images Strategy with cropped landing zones  
    run_prompt_test(
        image_path="landing_zones/",
        scenario="multi_cropped_comparison",
        strategies=["multi"],
        prompts=["prompt1", "envelope"],
        models=test_models
    )
    
    # Test 3: Both strategies with single annotated image for direct comparison
    run_prompt_test(
        image_path="images/flat_surfaces_annotated.jpg",
        scenario="strategy_comparison",
        strategies=["single", "multi"],
        prompts=["envelope"],  # Use JSON prompt for cleaner comparison
        models=test_models
    )
    
    print("\nPrompt testing framework ready!")
    print("Modify the parameters in the main section to run your specific tests.")
