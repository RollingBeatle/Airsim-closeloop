#This combines both monocular and stereo landing pipelines for a drone using AirSim.
# It captures images, overlays a grid, asks an LLM for a target cell, and
# computes the drone's position to land accurately on the target cell.

#LLM integration is left as a TODO, but hardcoded target cell is used for demonstration.


import cosysairsim as airsim
import numpy as np
import cv2
import time
import math
import openai
import torch
from PIL import Image
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

def ask_llm_for_cell(image_path):
    """TODO: integrate with your LLM (e.g., GPT-4V). Return label like 'C5'."""
    # TODO: Read image, send to LLM with prompt "Which grid cell A1–G7?"
    #       Parse and return the label string.
    raise NotImplementedError

def label_to_pixel(label):
    """Converts 'C5' → (row=2,col=4) → (px,py)."""
    row = ord(label[0]) - ord('A')
    col = int(label[1:]) - 1
    px = col*CELL_W + CELL_W//2
    py = row*CELL_H + CELL_H//2
    return px, py
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

# -------------------- MONOCULAR PIPELINE --------------------
def monocular_landing():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True); client.armDisarm(True)
    client.takeoffAsync().join(); time.sleep(1)
    z0 = -np.random.uniform(15, 25)
    client.moveToZAsync(z0,2).join(); time.sleep(1)

    # 1) capture and overlay
    resp = client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0]
    img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
    grid = overlay_grid(img)
    cv2.imwrite("mono_grid.png", cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))

    # 2) ask LLM for grid cell
    label = ask_llm_for_cell("mono_grid.png")
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
    # TODO: integrate MiDaS depth-estimate for accurate Z

# -------------------- STEREO PIPELINE --------------------
def stereo_landing():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True); client.armDisarm(True)
    client.takeoffAsync().join(); time.sleep(1)
    z0 = -np.random.uniform(15, 25)
    client.moveToZAsync(z0,2).join(); time.sleep(1)

    # capture left at pose1
    p1 = client.getMultirotorState().kinematics_estimated.position
    imgL = np.frombuffer(client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0].image_data_uint8, np.uint8).reshape(IMAGE_HEIGHT,IMAGE_WIDTH,3)

    # capture right at baseline
    client.moveToPositionAsync(p1.x_val, p1.y_val+BASELINE, p1.z_val,3).join(); time.sleep(3)
    imgR = np.frombuffer(client.simGetImages([airsim.ImageRequest(CAM_NAME,airsim.ImageType.Scene,False,False)])[0].image_data_uint8, np.uint8).reshape(IMAGE_HEIGHT,IMAGE_WIDTH,3)
    client.moveToPositionAsync(p1.x_val, p1.y_val, p1.z_val,3).join(); time.sleep(3)

    # overlay grid on left
    grid = overlay_grid(imgL)
    cv2.imwrite("stereo_grid.png", cv2.cvtColor(grid,cv2.COLOR_RGB2BGR))

    # ask LLM for cell
    label = ask_llm_for_cell("stereo_grid.png")
    px, py = label_to_pixel(label)

    # stereo disparity → depth at (px,py)
    grayL = cv2.cvtColor(imgL,cv2.COLOR_RGB2GRAY); grayR = cv2.cvtColor(imgR,cv2.COLOR_RGB2GRAY)
    stereo = cv2.StereoBM_create(numDisp=128, blockSize=15)
    disp = stereo.compute(grayL,grayR).astype(np.float32)/16.0; disp[disp<=0]=np.nan
    fx = IMAGE_WIDTH/(2*math.tan(math.radians(FOV_DEGREES/2)))
    d = np.nanmean(disp[max(py-1,0):py+2, max(px-1,0):px+2])
    Z = fx * BASELINE / d

    # back-project pixel to world
    cx, cy = IMAGE_WIDTH/2, IMAGE_HEIGHT/2
    x_cam = (px-cx)/fx * Z; y_cam = (py-cy)/fx * Z
    east = x_cam; north = -y_cam
    tx = p1.x_val + north; ty = p1.y_val + east; tz = p1.z_val + Z

    client.moveToPositionAsync(tx,ty,tz,3).join(); time.sleep(1)
    client.landAsync().join(); client.armDisarm(False)

# Example usage:
# monocular_landing()
# stereo_landing()
