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
    
        
    def crop_surfaces(self, area, img, alt_name=None, scale=None):
        # cropped_image = image[startY:endY, startX:endX]

        out = img.copy()
        i = 0
        new_areas = []
        for a in area:
            x_1, x_2, y_1, y_2 = a[1], a[3], a[0], a[2]
            w = x_2 - x_1
            h = y_2 - y_1
            cx = x_1 + w //2
            cy = y_1 + h //2
            
            fname = f'landing_zones/landing_zone{i}.jpg'
            if alt_name: 
                fname = f'landing_zones/test_landing_zone{alt_name}.jpg'
            if scale:
                new_w = int(w*scale)
                new_h = int(h*scale)

                new_x1 = max(cx - new_w //2, 0)
                new_y1 = max(cy - new_h //2, 0)

                new_x2 = min(cx + new_w //2, img.shape[1])
                new_y2 = min(cy + new_h //2, img.shape[0])

                marg_crop = out[new_y1:new_y2, new_x1:new_x2]
                new_areas.append((new_y1,new_x1,new_y2,new_x2))
                cv2.imwrite(fname, cv2.cvtColor(marg_crop,cv2.COLOR_RGB2BGR))
                # cv2.imwrite(fname,marg_crop)
            else:
                crop = out[y_1:y_2, x_1:x_2]
                
                cv2.imwrite(fname, cv2.cvtColor(crop,cv2.COLOR_RGB2BGR))
            i+=1
        if scale:
            return new_areas
        else: return area
    
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
    
    def segment_surfaces(self, img, original):
        # 1. Smooth the depth map (reduce noise)
        depth = cv2.GaussianBlur(img, (3, 3), 0)

        # 2. Compute gradient magnitude (surface slope)
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # 3. Threshold flat regions
        threshold = 25  # <-- tune this (try 15–35 depending on depth scale)
        flat_mask = (grad_mag < threshold).astype(np.uint8)

        # 4. Morphological filtering
        kernel = np.ones((3, 3), np.uint8)
        flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_OPEN, kernel)
        flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_CLOSE, kernel)

        # 5. Contour detection instead of regionprops
        contours, _ = cv2.findContours(flat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare annotation
        annotated = original.copy()
        areas = []
        height_src, width_src = img.shape
        size = height_src * width_src

        for c in contours:
            area = cv2.contourArea(c)
            if area > 700:  # filter small noise
                x, y, w, h = cv2.boundingRect(c)

                # avoid bounding box covering whole image
                if not (w * h == size):
                    # cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    areas.append((y, x, y + h, x + w))  # match (minr, minc, maxr, maxc)
        areas = self.filter_bb(areas)
        for area in areas:
            cv2.rectangle(annotated, (area[1], area[0]), (area[3], area[2]), (0, 255, 255), 2)
        # Save annotated result
        cv2.imwrite("images/flat_surfaces_annotated.jpg", annotated)

        return areas
    
    def containment(self, box1, box2):
        return (box1[0] <= box2[0] and
                box1[1] <= box2[1] and
                 box1[2] >= box2[2] and
                  box1[3] >= box2[3]  )
    
    def filter_bb(self, boxes):
        suitable = []
        for i, bb_i in enumerate(boxes):
            discard = False
            for j, bb_j in enumerate(boxes):
                if i != j and self.containment(bb_i,bb_j):
                    print("ymin,    xmin,   ymax,    xmax")
                    print(bb_i, "contains", bb_j)
                    discard = True
                    break
            if not discard:
                suitable.append(bb_i)
        return suitable

        # segment images based on depth map
    def segment_surfaces1(self, img, original):
        
        depth = cv2.GaussianBlur(img, (3, 3), 0)

        # Compute gradient magnitude
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold to get flat regions
        threshold = 0.1 * np.max(depth)
        print("the segmentation threshold is", threshold)
        kernel = np.ones((3,3), np.uint8)
        flat_mask = (grad_mag < threshold).astype(np.uint8)
        flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_OPEN, kernel)
        flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_CLOSE, kernel)

        dist = cv2.distanceTransform(flat_mask, cv2.DIST_L2, 5)
        _, markers = cv2.connectedComponents((dist > 0.5 *dist.max()).astype(np.uint8))
        markers = cv2.watershed(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), markers)
        
        # Label regions
        labeled = measure.label(flat_mask, connectivity=1)
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
                if not 0.95*size < maxc*maxr:
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
            print("The area size then image size", area_size, img_size)
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
    
    def bundle_crop_info(self, area, image, name):
        min_y, min_x, max_y, max_x = area
        crop = {
            "min_y": min_y,
            "min_x": min_x,
            "max_y": max_y,
            "max_x": max_x,
            "pil" : image,
            "name": name,
            "center_y":(max_y+min_y)//2,
            "center_x":(max_x+min_x)//2,
        }
        return crop

