import numpy as np
import cv2
import time
import math
import torch
from PIL import Image
from transformers import pipeline
from skimage import measure

class ImageProcessing:

    def __init__(self, width, height, fov_degrees, debug=True):
        # Camera settings
        self.width = width
        self.height = height
        self.fov_degrees = fov_degrees
        self.debug = debug
    
        
    def crop_surfaces(self, area, img, alt_name=None):
        out = img.copy()
        i = 0
        for a in area:
            crop = out[a[0]:a[2], a[1]:a[3]]
            if alt_name: 
                fname = f'landing_zones/test_landing_zone{alt_name}.jpg'
            else:
                fname = f'landing_zones/landing_zone{i}.jpg'
            cv2.imwrite(fname, cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
            i+=1
    
    def crop_five_cuadrants(self, image):

        detection = Image.open(image)
        w, h = detection.size
        left = (w - 480) // 2
        top = (h - 270) // 2
        right = (w + 480) // 2
        bottom = (h + 270) // 2
        ul_bb = (0,0,w//2,h//2)
        ur_bb = (w//2,0,w,h//2)
        bl_bb = (0,h//2,w//2,h)
        br_bb = (w//2,h//2,w,h)
        center_bb = (left,top,right,bottom)
        
        upper_left = detection.crop(ul_bb)
        upper_right = detection.crop(ur_bb)
        bottom_left = detection.crop(bl_bb)
        bottom_right = detection.crop(br_bb)
        center = detection.crop(center_bb)

        bounding_box = [ul_bb, ur_bb, bl_bb, br_bb, center_bb]
        detections = [upper_left, upper_right, bottom_left, bottom_right, center]
        if self.debug:
            # Show all detections
            for i in range(len(detections)):
                detections[i].show()
                input(f"Showing {i} instance, press enter")
                    
        return detections, bounding_box
    
        # Depth Anything V2
    def depth_analysis_depth_anything(self, image:Image):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load pipeline
        print("Running pipeline....")
        pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
        # inference
        depth_image = pipe(image)["depth"]
        depth_image.save("images/depth_image.jpg")
        if self.debug: 
            depth_image.show()
        return depth_image
    
        # segment images based on depth map
    def segment_surfaces(self, img, original):
        
        depth = cv2.GaussianBlur(img, (3, 3), 0)

        # Compute gradient magnitude
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold to get flat regions
        flat_mask = (grad_mag < 50).astype(np.uint8)
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
                # min_y min_x max_y max_x
                minr, minc, maxr, maxc = p.bbox
                cv2.rectangle(annotated, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
                if not 0.8*size < maxc*maxr:
                    print("percentage of area", (maxc*maxr)/size )
                    areas.append((minr, minc, maxr, maxc))       
        # Save annotated image
        
        cv2.imwrite("images/flat_surfaces_annotated.jpg", cv2.cvtColor(annotated,cv2.COLOR_RGB2BGR))
        if self.debug:
            input("Press to continue")
        return areas
    
    def match_areas(self, areas, select_pil_image):
        for area in areas:
            print("The area is", area)
            minr, minc, maxr, maxc = area

            area_size = (maxr - minr) * (maxc - minc)
            img_size = select_pil_image.size[0] * select_pil_image.size[1]

            if area_size == img_size:
                    print('ymin', minr, 'ymax', maxr)
                    print('xmin', minc, 'xmax', maxc)
                    px = (minc + maxc) // 2
                    py = (minr + maxr) // 2
                    print('center pixel', px, py)
                    break
        return px, py
    
    def inverse_perspective_mapping(self, pose, px, py, surface_height):
        A = surface_height
        hFOV = math.radians(self.fov_degrees)
        vFOV = 2*math.atan(math.tan(hFOV/2)*(self.height/self.width))
        world_w = 2*A*math.tan(hFOV/2); world_h = 2*A*math.tan(vFOV/2)
        mpp_x = world_w/self.width; mpp_y = world_h/self.height
        dx = px - self.width/2; dy = py - self.height/2
        north = -dy*mpp_y; east = dx*mpp_x
        tx = pose.x_val + north; ty = pose.y_val + east; tz = pose.z_val
        return tx,ty,tz

    def get_Z_Values_From_Depth_Map(self, surface_height, current_height, depth_map):
        depth_map = np.array(depth_map)
        print(depth_map.shape)
        ground_value = np.min(depth_map)
        surface_value = depth_map[self.height//2][self.width//2] 
  
        # Calculate slope
        m = (current_height - surface_height) / (ground_value - surface_value)
        # Calculate y-intercept
        b = surface_height - m * surface_value
        
        # Return a function y(x)
        return lambda x: m * x + b
    
    def inverse_perspective_mapping_v2(self, pose, px, py, surface_height, orientation):
        A = surface_height
        hFOV = math.radians(self.fov_degrees)
        vFOV = 2 * math.atan(math.tan(hFOV/2) * (self.height/self.width))
        world_w = 2 * A * math.tan(hFOV/2)
        world_h = 2 * A * math.tan(vFOV/2)
        mpp_x = world_w / self.width
        mpp_y = world_h / self.height

        dx = px - self.width/2
        dy = py - self.height/2

        # offsets in camera/local frame
        north = -dy * mpp_y
        east  = dx * mpp_x
        
        # import pdb; pdb.set_trace()
        # convert quaternion -> yaw
        q = orientation
        siny_cosp = 2 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy_cosp = 1 - 2 * (q.y_val**2 + q.z_val**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # rotate into world frame
        north_w =  math.cos(yaw) * north - math.sin(yaw) * east
        east_w  =  math.sin(yaw) * north + math.cos(yaw) * east

        # apply translation
        tx = pose.x_val + north_w
        ty = pose.y_val + east_w
        tz = pose.z_val

        return tx, ty, tz
