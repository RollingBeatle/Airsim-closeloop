#This combines both monocular and stereo landing pipelines for a drone using AirSim.
# It captures images, overlays a grid, asks an LLM for a target cell, and
# computes the drone's position to land accurately on the target cell.

#LLM integration is left as a TODO, but hardcoded target cell is used for demonstration.


import cosysairsim as airsim
import numpy as np
import cv2
import time
import math
import torch
from PIL import Image
from transformers import pipeline
from skimage import measure
# -------------------- CONFIG --------------------
IMAGE_WIDTH   = 960
IMAGE_HEIGHT  = 540
FOV_DEGREES   = 90
GRID_ROWS     = 7
GRID_COLS     = 7
CELL_W        = IMAGE_WIDTH  // GRID_COLS
CELL_H        = IMAGE_HEIGHT // GRID_ROWS
CAM_NAME      = "frontcamera"
BASELINE      = 2.0            # stereo baseline in meters
EXAMPLE_POS   = (0,-35,-100)
FIXED        = True        


# -------------------- HELPERS --------------------

def overlay_grid(img, highlight=None):
    """Draws a GRID_ROWS×GRID_COLS grid on img; highlight is (row,col) to box."""
    out = img.copy()
    for i in range(1, GRID_ROWS):
        cv2.line(out, (0, i*CELL_H), (IMAGE_WIDTH, i*CELL_H), (0,255,0), 2)
    for j in range(1, GRID_COLS):
        cv2.line(out, (j*CELL_W, 0), (j*CELL_W, IMAGE_HEIGHT), (0,255,0), 2)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            label = f"{chr(ord('A')+r)}{c+1}"
            cx, cy = c*CELL_W+CELL_W//2, r*CELL_H+CELL_H//2
            cv2.putText(out, label, (cx-20, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    if highlight:
        r,c = highlight
        x1,y1 = c*CELL_W, r*CELL_H
        x2,y2 = x1+CELL_W, y1+CELL_H
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,0,255),3) 
    
    
    return out

def label_to_pixel(label):
    """Converts 'C5' → (row=2,col=4) → (px,py)."""
    row = ord(label[0]) - ord('A')
    col = int(label[1:]) - 1
    px = col*CELL_W + CELL_W//2
    py = row*CELL_H + CELL_H//2
    return px, py

def crop_surfaces(area, img):
    out = img.copy()
    i = 0
    for a in area:
        crop = out[a[0]:a[2], a[1]:a[3]]
        fname = f'landing_zones/landing_zone{i}.jpg'
        cv2.imwrite(fname, crop)
        i+=1

# -------------------- DEPTH ESTIMATION -------------------
def midas_depth_estimate(image_path, px_c, py_c, cur_z):

    """Estimate depth using MiDaS model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Choose "DPT_Large" or "DPT_Hybrid" for higher-res output
    midas_model_type = "DPT_Hybrid"
    midas = torch.hub.load("intel-isl/MiDaS", midas_model_type)
    midas.to(device).eval()

    # Use appropriate transform
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if midas_model_type.startswith("DPT"):
        midas_transform = transforms.dpt_transform
    else:
        midas_transform = transforms.small_transform
        raise NotImplementedError
    
    img_pil    = Image.fromarray(img2)
    img_np     = np.array(img_pil)
    img_tensor = midas_transform(img_np)
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # Run MiDaS + interpolate
    with torch.no_grad():
        pred = midas(img_tensor)                        # (1, h_pred, w_pred)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),                          # -> (1,1,h_pred,w_pred)
            size=(IMAGE_HEIGHT, IMAGE_WIDTH),
            mode="bicubic",
            align_corners=False
        )  # -> (1,1,540,960)

    # Drop batch & channel dims
    depth_map = pred[0, 0].cpu().numpy()                # -> (540,960)

    # Now safe to index
    cell_depth = float(depth_map[py_c, px_c])
    print(f"[MiDaS] Depth at cell ({px_c},{py_c}): {cell_depth:.3f}")

    # Estimate landing Z in NED (cur_z negative up)
    landing_z = cur_z + cell_depth
    print(f"→ Descending to estimated ground Z = {landing_z:.2f}")
    return landing_z

# Depth Anything V2
def depth_analysis_depth_anything(image:Image):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load pipeline
    print("Running pipeline....")
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    # inference
    depth_image = pipe(image)["depth"]
    depth_image.save("depth_image.jpg")
    return depth_image

# segment images based on depth map
def segment_surfaces(img, original):
    
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

def get_z_value(client:airsim.MultirotorClient, depth_map, area):
    # get the current altitude of the drone
    altitude = -client.getMultirotorState().kinematics_estimated.position.z_val
    #.gps_location.altitude
    depth_img = np.array(depth_map)
    
    depth_inverted = 255.0 - depth_img
 
    x, y = get_farthest_point(depth_inverted)
    value_far = depth_img[y,x]
    # TODO: this should be a more specific way of getting the exact landing zone
    # img = cv2.imread(f"landing_zones/"+selected_surface)
    # a,b = img.shape[:2]
    # x_surface, y_surface = a//2, b//2
    # find the center of the bounding box
    sample = area[0]
    y_center, x_center  = sample[2]//2, sample[3]//2
   
    scaling_factor = altitude/value_far
    converted_map = depth_inverted*scaling_factor
    value_area = converted_map[y_center,x_center]
    print("Min depth:", depth_inverted.min())
    print("Max depth:", depth_inverted.max())
    print("Depth at farthest point:", value_far)
    print(f"real distance is about {value_area} meters")
    # This is a rough fix
    return -(altitude - (value_area*2)) 



def get_farthest_point(depth_img):
    
    # This is a big assumption and needs further discussion but we are
    # assuming the furthest point in the depth map is the ground
    # depth_img = np.array(depth_img) 
    # the size of the patch we are looking for
    floor_patch = 500
    # go through the numpy array from the min value until pitch black
    for i in range(int(depth_img.min()), 255):
        # find the darkest path of size 500
        if np.count_nonzero(depth_img==i) >= floor_patch:
            b = np.where(depth_img==i, 1, 0).astype(np.uint8)
            break

    # separate the section
    contours,_ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = max(contours, key=cv2.contourArea)
    m = cv2.moments(max_contour)

    # Get bounding box
    x_, y_, w, h = cv2.boundingRect(max_contour) 
    center_x = int(m["m10"] / m["m00"])
    center_y = int(m["m01"] / m["m00"])

    return center_x, center_y
    
    
# -------------------- MONOCULAR PIPELINE --------------------
def monocular_landing(llm_call, position):

    client = airsim.MultirotorClient()
    if position:
        position_drone(client)
    # 1) capture and overlay
    resp = client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
    img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
    pillow_img = Image.fromarray(img)
    # depth map image and segmentation
    depth_map = depth_analysis_depth_anything(pillow_img)
    img2 = np.array(depth_map)
    # get boxes of surfaces
    areas = segment_surfaces(img2, np.array(pillow_img))
    # crop
    crop_surfaces(areas, img)
    # get Z distance and move, this is temp
    # TODO: move this to after we get the desired square
    z_distance = int(get_z_value(client,depth_map, areas))
    client.moveToZAsync(z_distance,3).join()
    
    grid = overlay_grid(img)
    cv2.imwrite("images/mono_grid.jpg", cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))
    
    # 2) ask LLM for grid cell
    label = llm_call("images/mono_grid.jpg")
    px, py = label_to_pixel(label)

    # 3) pixel→world XY via IPM
    pose = client.getMultirotorState().kinematics_estimated.position
    A = abs(pose.z_val)
    hFOV = math.radians(FOV_DEGREES)
    vFOV = 2*math.atan(math.tan(hFOV/2)*(IMAGE_HEIGHT/IMAGE_WIDTH))
    world_w = 2*A*math.tan(hFOV/2); world_h = 2*A*math.tan(vFOV/2)
    mpp_x = world_w/IMAGE_WIDTH; mpp_y = world_h/IMAGE_HEIGHT
    dx = px - IMAGE_WIDTH/2; dy = py - IMAGE_HEIGHT/2
    north = -dy*mpp_y; east = dx*mpp_x
    tx = pose.x_val + north; ty = pose.y_val + east; tz = pose.z_val

    client.moveToPositionAsync(tx,ty,tz,3).join(); time.sleep(1)

    # 4) (Optional) MiDaS descent, omitted here
    # TODO: DO integrate MiDaS depth-estimate for accurate Z

# -------------------- STEREO PIPELINE --------------------
def stereo_landing(llm_call, position):
    client = airsim.MultirotorClient()
    if position:
        position_drone(client)
    # capture left at pose1
    p1 = client.getMultirotorState().kinematics_estimated.position
    imgL = np.frombuffer(client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0].image_data_uint8, np.uint8).reshape(IMAGE_HEIGHT,IMAGE_WIDTH,3)

    # capture right at baseline
    client.moveToPositionAsync(p1.x_val, p1.y_val+BASELINE, p1.z_val,3).join(); time.sleep(3)
    imgR = np.frombuffer(client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0].image_data_uint8, np.uint8).reshape(IMAGE_HEIGHT,IMAGE_WIDTH,3)
    client.moveToPositionAsync(p1.x_val, p1.y_val, p1.z_val,3).join(); time.sleep(3)

    # overlay grid on left
    grid = overlay_grid(imgL)
    cv2.imwrite("images/stereo_grid.jpg", cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))

    # ask LLM for cell
    label = llm_call("images/stereo_grid.jpg")
    px, py = label_to_pixel(label)

    # stereo disparity → depth at (px,py)
    grayL = cv2.cvtColor(imgL,cv2.COLOR_RGB2GRAY); grayR = cv2.cvtColor(imgR,cv2.COLOR_RGB2GRAY)
    stereo = cv2.StereoBM_create(128, 15)
    disp = stereo.compute(grayL,grayR).astype(np.float32)/16.0; disp[disp<=0]=np.nan
    fx = IMAGE_WIDTH/(2*math.tan(math.radians(FOV_DEGREES/2)))
    d = np.nanmean(disp[max(py-1,0):py+2, max(px-1,0):px+2])
    Z = fx * BASELINE / d

    # back-project pixel to world
    cx, cy = IMAGE_WIDTH/2, IMAGE_HEIGHT/2
    x_cam = (px-cx)/fx * Z; y_cam = (py-cy)/fx * Z
    east = x_cam; north = -y_cam
    tx = float(p1.x_val + north); ty = float(p1.y_val + east); tz = float(p1.z_val + Z)
    print(tx,ty,tz)
    client.moveToPositionAsync(tx,ty,tz,3).join(); time.sleep(1)
    # client.landAsync().join(); client.armDisarm(False)

# -------------------- MOVEMENT ---------------------------
def position_drone(client:airsim.MultirotorClient):
    # Position the drone randomly in demo
    client.confirmConnection()
    client.enableApiControl(True); client.armDisarm(True)
    client.takeoffAsync().join(); time.sleep(1)
    if FIXED:
        x,y,z = EXAMPLE_POS
        client.moveToPositionAsync(x,y,z,3).join(); time.sleep(1)
    else:
        z0 = -np.random.uniform(40, 50)
        client.moveToZAsync(z0,2).join(); time.sleep(1)
    
def land_drone(client:airsim.MultirotorClient, x, y ,z): 

    client.moveToPositionAsync(x,y,z,3).join(); time.sleep(1)
    client.landAsync().join(); client.armDisarm(False)


if __name__ == "__main__":
    stereo_landing()
    # monocular_landing()

