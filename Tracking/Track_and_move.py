import cosysairsim as airsim
import cv2
import numpy as np
import time
from PIL import Image, ImageEnhance
import io

def extract_patch(image, center, size):
    x, y = center
    half = size // 2
    h, w = image.shape[:2]
    x1, x2 = max(0, x - half), min(w, x + half)
    y1, y2 = max(0, y - half), min(h, y + half)
    return image[y1:y2, x1:x2]

def apply_gamma_correction(image, gamma=1.5):
    """
    Apply gamma correction for exposure adjustment.
    gamma > 1  darker
    gamma < 1  brighter
    """
    img_float = image.astype(np.float32) / 255.0
    img_gamma = np.power(img_float, gamma)
    img_out = np.clip(img_gamma * 255, 0, 255).astype(np.uint8)
    return img_out

def get_current_image(client):
    response = client.simGetImage("bottom_center", airsim.ImageType.Scene)
    image = Image.open(io.BytesIO(response))

    # Convert image to numpy array (PIL gives RGB)
    image_array = np.array(image)

    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Normalize pixel values to [0, 1] for gamma correction
    image_array_normalized = image_array / 255.0

    # Gamma correction (exposure reduction)
    exposure_reduction_factor = 4  # >1 darkens the image; increase for more reduction
    image_gamma_corrected = np.power(image_array_normalized, exposure_reduction_factor)

    # Scale back to [0, 255]
    image_gamma_corrected = np.clip(image_gamma_corrected * 255, 0, 255).astype(np.uint8)

    return image_gamma_corrected


def track_template(landing_zone, current_image, landing_center=None, search_radius=None):
    h_img, w_img = current_image.shape[:2]
    h_patch, w_patch = landing_zone.shape[:2]

    if landing_center is None:
        landing_center = (w_img // 2, h_img // 2)

    if search_radius is not None:
        cx, cy = landing_center
        x1 = max(0, cx - search_radius)
        y1 = max(0, cy - search_radius)
        x2 = min(w_img, cx + search_radius + w_patch)
        y2 = min(h_img, cy + search_radius + h_patch)
        search_region = current_image[y1:y2, x1:x2]
        offset = (x1, y1)
    else:
        search_region = current_image
        offset = (0, 0)

    result = cv2.matchTemplate(search_region, landing_zone, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    top_left = (max_loc[0] + offset[0], max_loc[1] + offset[1])
    new_center = (top_left[0] + w_patch // 2, top_left[1] + h_patch // 2)

    dx = new_center[0] - landing_center[0]
    dy = new_center[1] - landing_center[1]

    return (dx, dy), new_center


def move_in_direction(client, dx_pixels, dy_pixels, step_distance_meters=1):
    # Calculate direction vector from dx, dy
    vec = np.array([dx_pixels, dy_pixels])

    norm = np.linalg.norm(vec)
    if norm == 0:
        print("No movement needed.")
        return

    # Normalize direction vector
    direction = vec / norm

    # Apply fixed step in NED coordinates
    ned_x_offset = -direction[1] * step_distance_meters
    ned_y_offset = direction[0] * step_distance_meters

    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position

    target_x = pos.x_val + ned_x_offset
    target_y = pos.y_val + ned_y_offset
    target_z = pos.z_val  # Keep altitude fixed

    print(f"Moving to: X={target_x:.2f}, Y={target_y:.2f}, Z={target_z:.2f}")
    client.moveToPositionAsync(target_x, target_y, target_z, velocity=1.0).join()
    print("FinishedMovement")

def land(client):
    distance_sensor_data = client.getDistanceSensorData("Distance", "Drone1")
    distance = distance_sensor_data.distance
    print(f"Distance to ground: {distance:.2f}")
    target_z = distance
    client.moveToZAsync(target_z, 2).join() 
    client.landAsync().join()

def main_loop():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()
    client.moveToZAsync(-50, 2).join() 

    image = get_current_image(client)
    image = get_current_image(client)
    h_img, w_img = image.shape[:2]

    # This will change to be our actual landing zone for now im just setting it to some spot in the upper left hand area
    landing_center = (int(w_img * 0.40), int(h_img * 0.40))
    print(f"Initial landing zone center: {landing_center}")

    patch = extract_patch(image, landing_center, size=100)
    cv2.imwrite("initial_full_image.png", image)
    cv2.imwrite("initial_landing_patch.png", patch)
    with open("initial_landing_center.txt", "w") as f:
        f.write(f"{landing_center[0]},{landing_center[1]}\n")
    print("Initial landing zone data saved to 'initial_landing_patch.png' and 'initial_landing_center.txt'.")

    threshold_pixels = 8

    while True:
        current_image = get_current_image(client)
        image_center = (w_img // 2, h_img // 2)

        (dx, dy), predicted_center = track_template(patch, current_image, landing_center, search_radius=250)
        dx_center_to_landing = predicted_center[0] - image_center[0]
        dy_center_to_landing = predicted_center[1] - image_center[1]

        print(f"Predicted center: {predicted_center}")
        print(f"[Step] dx: {dx_center_to_landing}, dy: {dy_center_to_landing}")

        if abs(dx_center_to_landing) < threshold_pixels and abs(dy_center_to_landing) < threshold_pixels:
            print("Landing zone centered. Ready for descent.")
            break

        # move_drone_fixed_speed(client, dx_center_to_landing, dx_center_to_landing, speed=1.0)
        move_in_direction(client, dx_center_to_landing, dy_center_to_landing)

        landing_center = predicted_center
        patch = extract_patch(current_image, landing_center, size=100)

    land(client)

    client.armDisarm(False)
    client.enableApiControl(False)


if __name__ == "__main__":
    main_loop()
