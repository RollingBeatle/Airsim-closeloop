import cosysairsim as airsim
import numpy as np
import cv2
import time
import math
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

# 0-indexed target cell (e.g. F7 => row=5, col=6)
TARGET_ROW    = 0
TARGET_COL    = 4

CAM_NAME      = "frontcamera"    # downward-facing camera name in your settings.json

# -------------------- LOAD MiDaS MODEL --------------------
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

# -------------------- AIRSIM CLIENT SETUP --------------------
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# -------------------- TAKEOFF & ASCEND --------------------
print("🚁 Taking off...")
client.takeoffAsync().join()
time.sleep(1)

rand_alt = -np.random.uniform(15, 25)
print(f"🚁 Ascending to Z = {rand_alt:.1f}...")
client.moveToZAsync(rand_alt, 2).join()
time.sleep(1)

# -------------------- HORIZONTAL MOVE (XY MAPPING) --------------------
# Capture initial downward image
resp = client.simGetImages([airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene, False, False)])[0]
img1d  = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
img_rgb = img1d.reshape(resp.height, resp.width, 3)


# Overlay grid lines
grid_img = img_rgb.copy()
for i in range(1, GRID_ROWS):
    y = i * CELL_H
    cv2.line(grid_img, (0, y), (IMAGE_WIDTH, y), (0, 255, 0), 2)
for j in range(1, GRID_COLS):
    x = j * CELL_W
    cv2.line(grid_img, (x, 0), (x, IMAGE_HEIGHT), (0, 255, 0), 2)

# Draw cell labels
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        label = f"{chr(ord('A') + row)}{col+1}"
        cx = col * CELL_W + CELL_W // 2
        cy = row * CELL_H + CELL_H // 2
        cv2.putText(grid_img, label, (cx - 20, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Highlight the target cell
x1, y1 = TARGET_COL * CELL_W, TARGET_ROW * CELL_H
x2, y2 = x1 + CELL_W, y1 + CELL_H
cv2.rectangle(grid_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

# Save the overlaid image
cv2.imwrite("xy_initial_grid.png", cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
print("💾 Saved xy_initial_grid.png with grid overlay")

# Get current pose
pose = client.getMultirotorState().kinematics_estimated.position
cur_x, cur_y, cur_z = pose.x_val, pose.y_val, pose.z_val
altitude = abs(cur_z)

# Compute FOV and meters-per-pixel
hFOV = math.radians(FOV_DEGREES)
vFOV = 2 * math.atan(math.tan(hFOV/2) * (IMAGE_HEIGHT/IMAGE_WIDTH))
world_w = 2 * altitude * math.tan(hFOV/2)
world_h = 2 * altitude * math.tan(vFOV/2)
mpp_x = world_w / IMAGE_WIDTH
mpp_y = world_h / IMAGE_HEIGHT

# Compute pixel center of target
px_c = TARGET_COL * CELL_W + CELL_W // 2
py_c = TARGET_ROW * CELL_H + CELL_H // 2

dx_px = px_c - (IMAGE_WIDTH / 2)
dy_px = py_c - (IMAGE_HEIGHT / 2)

east_offset  = dx_px * mpp_x
north_offset = -dy_px * mpp_y

target_x = cur_x + north_offset
target_y = cur_y + east_offset
target_z = cur_z

print(f"→ Flying horizontally to (x={target_x:.2f}, y={target_y:.2f}, z={target_z:.2f})")
client.moveToPositionAsync(target_x, target_y, target_z, 3).join()
time.sleep(1)

# Capture arrival image for MiDaS
resp2 = client.simGetImages([airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene, False, False)])[0]
img2 = np.frombuffer(resp2.image_data_uint8, dtype=np.uint8).reshape(resp2.height, resp2.width, 3)
cv2.imwrite("xy_arrival.png", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

# -------------------- MiDaS-BASED Z ESTIMATION --------------------
print("🔍 Processing image with MiDaS...")

# Prepare tensor
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

# -------------------- DESCEND & LAND --------------------
client.moveToPositionAsync(target_x, target_y, landing_z, 1).join()
time.sleep(1)
print("🛬 Landing...")
client.landAsync().join()
client.armDisarm(False)
print("✅ Touchdown complete!")
