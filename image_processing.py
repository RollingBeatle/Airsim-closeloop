import numpy as np
import cv2
import time
import math
import torch
from PIL import Image
from transformers import pipeline
from skimage import measure

class ImageProcessing:

    def __init__(self, width, height, fov_degrees):
        # Camera settings
        self.width = width
        self.height = height
        self.fov_degrees = fov_degrees
    
        
    def crop_surfaces(self, area, img):
        out = img.copy()
        i = 0
        for a in area:
            crop = out[a[0]:a[2], a[1]:a[3]]
            fname = f'landing_zones/landing_zone{i}.jpg'
            cv2.imwrite(fname, crop)
            i+=1
    
    def crop_five_cuadrants(self, image):

        detection = Image.fromarray(cv2.imread(image))
        w, h = detection.size
        upper_left = detection.crop((0,0,w//2,h//2))
        upper_right = detection.crop((w//2,0,w,h//2))
        bottom_left = detection.crop((0,h//2,w//2,h))
        bottom_right = detection.crop((w//2,h//2,w,h))
        
        print(f"The bounding box upper_left {upper_left}")
        print(f"The bounding box upper_right {upper_right}")
        print(f"The bounding box bottom_left {bottom_right}")
        print(f"The bounding box bottom_left {bottom_left}")

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
        return detections
    
        # Depth Anything V2
    def depth_analysis_depth_anything(self, image:Image):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load pipeline
        print("Running pipeline....")
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        # inference
        depth_image = pipe(image)["depth"]
        depth_image.save("images/depth_image.jpg")
        return depth_image
    
        # segment images based on depth map
    def segment_surfaces(self, img, original):
        
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
    
    def match_areas(self, areas, select_pil_image):
        for area in areas:
            print(area)
            area_size = (area[3]-area[1])*(area[2]-area[0])
            img_size = select_pil_image.size[0]*select_pil_image.size[1] 
            if area_size == img_size:
                px = ((area[3] + area[1])//2) 
                py = ((area[2] + area[0])//2) 
                print(px,py)
                break
        return px, py
    
    def inverse_perspective_mapping(self, pose, px, py):
        A = abs(pose.z_val)
        hFOV = math.radians(self.fov_degrees)
        vFOV = 2*math.atan(math.tan(hFOV/2)*(self.height/self.width))
        world_w = 2*A*math.tan(hFOV/2); world_h = 2*A*math.tan(vFOV/2)
        mpp_x = world_w/self.width; mpp_y = world_h/self.height
        dx = px - self.width/2; dy = py - self.height/2
        north = -dy*mpp_y; east = dx*mpp_x
        tx = pose.x_val + north; ty = pose.y_val + east; tz = pose.z_val
        return tx,ty,tz
