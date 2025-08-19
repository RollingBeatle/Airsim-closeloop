import numpy as np
import cv2
import time
import math
import torch
from PIL import Image
from transformers import pipeline, DepthAnythingConfig, DepthAnythingForDepthEstimation, AutoImageProcessor, AutoModelForDepthEstimation
from skimage import measure
import os

class ImageProcessing:

    def __init__(self, width, height, fov_degrees, debug=True):
        # Camera settings
        self.width = width
        self.height = height
        self.fov_degrees = fov_degrees
        self.debug = debug
        self.eps = 1e-6
    
        
    def crop_surfaces(self, area, img):
        out = img.copy()
        i = 0
        for a in area:
            crop = out[a[0]:a[2], a[1]:a[3]]
            fname = os.path.join('landing_zones', f"landing_zone{i}.jpg")
            cv2.imwrite(fname, crop)
            i+=1
    
    def crop_five_cuadrants(self, image):

        detection = Image.fromarray(cv2.imread(image))
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
    
    

    def depth_analysis_depth_anything(
        self,
        image: Image,
        save_path="images",
        debug=False,
        max_depth: float = 80.0  # set max depth in meters
    ):
        """
        Run DepthAnythingV2 in metric mode with configurable max_depth.
        Returns:
        - depth_raw: np.ndarray (depth in meters)
        - depth_vis: PIL.Image (grayscale visualization)
        """
        print(max_depth)
        max_depth = abs(max_depth)
        # import pdb; pdb.set_trace()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Load processor ---
        image_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf"
        )

        # --- Load model with custom config (metric + max_depth) ---
        config = DepthAnythingConfig.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf",
            depth_estimation_type="metric",
            max_depth=max_depth
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Small-hf",
            config=config
        ).to(device)

        # --- Prepare image for model ---
        inputs = image_processor(images=image, return_tensors="pt").to(device)

        # --- Run inference ---
        with torch.no_grad():
            outputs = model(**inputs)

        # --- Postprocess: resize back to original image size ---
        post_processed = image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        predicted_depth = post_processed[0]["predicted_depth"]  # torch.Tensor [H, W]

        # --- Raw depth in meters ---
        depth_raw = predicted_depth.detach().cpu().numpy()

        # --- Visualization (normalize 0–255) ---
        depth_norm = (depth_raw - depth_raw.min()) / (depth_raw.max() - depth_raw.min())
        depth_vis = Image.fromarray((depth_norm * 255).astype("uint8"))

        # --- Save visualization ---
        os.makedirs(save_path, exist_ok=True)
        depth_vis.save(os.path.join(save_path, "depth_image.jpg"))

        if debug:
            depth_vis.show()

        return depth_raw, depth_vis
    
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
                minr, minc, maxr, maxc = p.bbox
                cv2.rectangle(annotated, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
                if not size == maxc*maxr:
                    areas.append((minr, minc, maxr, maxc))       
        # Save annotated image
        cv2.imwrite(os.path.join('images', "flat_surfaces_annotated.jpg"), annotated)
        if self.debug:
            input("Press to continue")
        return areas
    
    def find_rotated_landing_zones(depth_map, window_size=10, flat_tol=0.1, min_area=100, merge_overlap_thresh=0.5):
        """
        Detect flat landing zones as rotated rectangles from a metric depth map.

        Parameters
        ----------
        depth_map : np.ndarray (H x W)
            Depth map in meters.
        window_size : int
            Size of the patch to initially check for flatness.
        flat_tol : float
            Maximum variation in depth to consider a patch flat.
        min_area : int
            Minimum area (in pixels) of a landing zone.
        merge_overlap_thresh : float
            Minimum fraction of edge overlap to allow merging adjacent rectangles.

        Returns
        -------
        landing_zones : list of cv2.RotatedRect
            Each element is ((cx, cy), (w, h), angle) in OpenCV RotatedRect format.
        """
        H, W = depth_map.shape

        # Step 1: Threshold for flat regions using sliding window
        flat_mask = np.zeros_like(depth_map, dtype=np.uint8)
        for i in range(0, H - window_size + 1, window_size):
            for j in range(0, W - window_size + 1, window_size):
                patch = depth_map[i:i+window_size, j:j+window_size]
                if np.ptp(patch) <= flat_tol:  # max-min < tolerance
                    flat_mask[i:i+window_size, j:j+window_size] = 255

        # Step 2: Find contours of flat regions
        contours, _ = cv2.findContours(flat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 3: Fit rotated rectangles and filter by area
        landing_zones = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            rect = cv2.minAreaRect(cnt)  # ((cx, cy), (w, h), angle)
            landing_zones.append(rect)

        # Step 4: Merge rectangles that share enough of an edge
        merged = True
        while merged:
            merged = False
            new_zones = []
            used = [False] * len(landing_zones)

            for i, rect1 in enumerate(landing_zones):
                if used[i]:
                    continue
                r1_pts = cv2.boxPoints(rect1)
                merged_rect = rect1

                for j, rect2 in enumerate(landing_zones):
                    if i == j or used[j]:
                        continue
                    r2_pts = cv2.boxPoints(rect2)

                    # Check if rectangles share a sufficient edge (intersection over min edge length)
                    inter_area = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
                    if inter_area is not None:
                        inter_area = cv2.contourArea(inter_area)
                        min_area_rect = min(cv2.contourArea(r1_pts), cv2.contourArea(r2_pts))
                        if inter_area / min_area_rect >= merge_overlap_thresh:
                            # Merge by computing bounding rectangle over all points
                            all_pts = np.vstack((r1_pts, r2_pts))
                            merged_rect = cv2.minAreaRect(all_pts)
                            used[j] = True
                            merged = True

                new_zones.append(merged_rect)
                used[i] = True

            landing_zones = new_zones

        return landing_zones
    

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
    
    # def inverse_perspective_mapping(self, pose, px, py, surface_height):
    #     A = surface_height
    #     hFOV = math.radians(self.fov_degrees)
    #     vFOV = 2*math.atan(math.tan(hFOV/2)*(self.height/self.width))
    #     world_w = 2*A*math.tan(hFOV/2); 
    #     world_h = 2*A*math.tan(vFOV/2)
    #     mpp_x = world_w/self.width; 
    #     mpp_y = world_h/self.height
    #     dx = px - self.width/2 
    #     dy = py - self.height/2
    #     north = -dy*mpp_y; 
    #     east = dx*mpp_x
    #     tx = pose.x_val + north; 
    #     ty = pose.y_val + east; 
    #     tz = pose.z_val
    #     import pdb; pdb.set_trace()
    #     print(f"current coords: {pose.x_val}, {pose.y_val}, {pose.z_val}")
    #     print(f"next coords: {tx}, {ty}, {tz}")
    #     return tx,ty,tz
    def inverse_perspective_mapping(self, pose, px, py, surface_height, orientation):
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
        
        import pdb; pdb.set_trace()
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

    def get_Z_Values_From_Depth_Map(self, surface_height, current_height, depth_map):
        depth_map = np.array(depth_map)
        print(depth_map)
        ground_value = np.min(depth_map)
        print(ground_value)
        surface_value = depth_map[self.width//2][self.height//2] 
  
        # Calculate slope
        m = (current_height - surface_height) / (ground_value - surface_value)
        # Calculate y-intercept
        b = surface_height - m * surface_value
        
        # Return a function y(x)
        return lambda x: m * x + b
    
    def get_Z_Values_From_Depth_Map_2(self, surface_height, depth_map, curr_height):
        """
        Assumes DepthAnythingV2-style output (inverse-ish depth).
        Uses the center pixel's raw value as the anchor (rangefinder reading = surface_height).
        
        Parameters
        ----------
        surface_height : float
            Rangefinder measurement (meters) to the point directly below (center pixel).
        depth_map : array_like, shape (H, W)
            Raw depth map produced by DepthAnythingV2 (NOT display-normalized 0..255 image).
        
        Returns
        -------
        func, s
            func(R_raw) -> Z_meters (callable that accepts scalar or numpy array),
            s : scale used in Z = s / (R + eps)
        """
        depth_map = np.asarray(depth_map, dtype=float)
        # import pdb; pdb.set_trace()
        if depth_map.shape != (self.height, self.width):
            # allow both (H,W) and (W,H) confusion - but prefer H,W
            # raise for clarity if shape mismatch
            raise ValueError(f"depth_map shape {depth_map.shape} != (height,width)=({self.height},{self.width})")

        # raw center pixel value (model's output at center)
        Rc = depth_map[self.height // 2, self.width // 2]

        # handle NaN or invalid center values
        if not np.isfinite(Rc) or abs(Rc) < self.eps:
            # center value is unusable; cannot form scale reliably
            raise ValueError(f"Center raw depth value is invalid ({Rc}). Ensure the model output and alignment are correct.")

        # For DepthAnythingV2 we assume inverse-like output: Z = s / R
        s = surface_height / Rc  # since Zc = surface_height ≈ s / Rc  => s ≈ Zc * Rc

        def to_Z(R_raw):
            """
            Map raw DepthAnythingV2 output(s) to metric distances.
            Accepts scalar, numpy array, or torch-like numeric arrays (compatible with numpy).
            """
            R = np.asarray(R_raw, dtype=float)
            # avoid divide-by-zero and very small values
            Z = s * R
            # clip to plausible range
            Z = np.clip(Z, 0.0, 148)
            return Z

        return to_Z, s
