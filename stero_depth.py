import cosysairsim as airsim
import numpy as np
import cv2
import time
import math

# -------------------- CONFIG --------------------
IMAGE_WIDTH   = 960
IMAGE_HEIGHT  = 540
FOV_DEGREES   = 90

GRID_ROWS     = 7
GRID_COLS     = 7
CELL_W        = IMAGE_WIDTH  // GRID_COLS
CELL_H        = IMAGE_HEIGHT // GRID_ROWS

# 0-indexed target cell (e.g. C5 => row=2, col=4)
TARGET_ROW    = 2
TARGET_COL    = 4

CAM_NAME      = "frontcamera"  # downward-facing camera

# Stereo settings
BASELINE      = 2.0            # meters between the two shots
numDisp       = 128            # must be divisible by 16
blockSize     = 15

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
print(f"🚁 Ascending to Z = {rand_alt:.1f} m")
client.moveToZAsync(rand_alt, 2).join()
time.sleep(1)

# Record first pose
pose1 = client.getMultirotorState().kinematics_estimated.position
x1, y1, z1 = pose1.x_val, pose1.y_val, pose1.z_val
print(f"[POSE1] x={x1:.2f}, y={y1:.2f}, z={z1:.2f}")

# -------------------- CAPTURE LEFT IMAGE --------------------
resp1 = client.simGetImages([airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene, False, False)])[0]
img1d = np.frombuffer(resp1.image_data_uint8, dtype=np.uint8)
imgL = img1d.reshape(resp1.height, resp1.width, 3)
cv2.imwrite("stereo_left.png", cv2.cvtColor(imgL, cv2.COLOR_RGB2BGR))
print("💾 Saved stereo_left.png")

# -------------------- MOVE BASELINE & CAPTURE RIGHT IMAGE --------------------
print(f"🚁 Moving {BASELINE}m east for stereo baseline...")
client.moveToPositionAsync(x1, y1 + BASELINE, z1, 3).join()
time.sleep(3)  # allow images to stabilize
resp2 = client.simGetImages([airsim.ImageRequest(CAM_NAME, airsim.ImageType.Scene, False, False)])[0]
img2d = np.frombuffer(resp2.image_data_uint8, dtype=np.uint8)
imgR = img2d.reshape(resp2.height, resp2.width, 3)
cv2.imwrite("stereo_right.png", cv2.cvtColor(imgR, cv2.COLOR_RGB2BGR))
print("💾 Saved stereo_right.png")

# -------------------- RETURN TO POSE1 --------------------
print("🚁 Returning to original XY...")
client.moveToPositionAsync(x1, y1, z1, 3).join()
time.sleep(3)  # allow drone to settle

# -------------------- STEREO DISPARITY & DEPTH --------------------
grayL = cv2.cvtColor(imgL, cv2.COLOR_RGB2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_RGB2GRAY)
stereo = cv2.StereoBM_create(numDisp, blockSize)
disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0  # disparity in pixels

# Filter invalid disparities
disp[disp <= 0] = np.nan

# Depth map
fx = IMAGE_WIDTH / (2 * math.tan(math.radians(FOV_DEGREES / 2)))
depth_map = fx * BASELINE / disp

# -------------------- GRID OVERLAY ON LEFT IMAGE --------------------
grid_img = imgL.copy()
for i in range(1, GRID_ROWS):
    cv2.line(grid_img, (0, i*CELL_H), (IMAGE_WIDTH, i*CELL_H), (0,255,0), 2)
for j in range(1, GRID_COLS):
    cv2.line(grid_img, (j*CELL_W, 0), (j*CELL_W, IMAGE_HEIGHT), (0,255,0), 2)
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        label = f"{chr(ord('A')+row)}{col+1}"
        cx = col*CELL_W + CELL_W//2
        cy = row*CELL_H + CELL_H//2
        cv2.putText(grid_img, label, (cx-20, cy+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
px_c = TARGET_COL*CELL_W + CELL_W//2
py_c = TARGET_ROW*CELL_H + CELL_H//2
cv2.rectangle(grid_img,
              (px_c-CELL_W//2, py_c-CELL_H//2),
              (px_c+CELL_W//2, py_c+CELL_H//2),
              (0,0,255), 3)
cv2.imwrite("grid_overlay.png", cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR))
print("💾 Saved grid_overlay.png")

# -------------------- COMPUTE TARGET WORLD COORDS --------------------
disp_val = np.nanmean(disp[max(py_c-1,0):py_c+2, max(px_c-1,0):px_c+2])
Z = fx * BASELINE / disp_val
print(f"[DEPTH] Mean disparity at C5: {disp_val:.2f}px -> Z={Z:.2f}m")

# Back-project pixel
cx = IMAGE_WIDTH/2
cy = IMAGE_HEIGHT/2
x_cam = (px_c - cx)/fx * Z
y_cam = (py_c - cy)/fx * Z

east_offset  = x_cam
north_offset = -y_cam

target_x = float(x1 + north_offset)
target_y = float(y1 + east_offset)
target_z = float(z1 + Z)
print(f"[TARGET] Flying to X:{target_x:.2f}, Y:{target_y:.2f}, Z:{target_z:.2f}")

# -------------------- FLY & LAND --------------------
client.moveToPositionAsync(target_x, target_y, target_z, 3).join()
time.sleep(1)
print("🛬 Landing...")
client.landAsync().join()
client.armDisarm(False)
print("✅ Landed at chosen stereo point (C5)!")
