# Lidar baseline
import cosysairsim as airsim
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import io
from PIL import Image, ImageEnhance
import cv2
from collections import deque
import math

"""
Fly drone up
get lidar data + take photo. 
Pick random landing spot
Go to it
Land

"""
def rotate_point_cloud(pcd, angle_deg=-90):
    # Convert the angle to radians
    angle_rad = np.radians(angle_deg)
    
    # Create the 2D rotation matrix for a clockwise rotation around the Z-axis (counterclockwise -90 degrees)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                [np.sin(angle_rad), np.cos(angle_rad), 0],
                                [0, 0, 1]])

    # Apply the rotation matrix to each point in the point cloud
    points = np.asarray(pcd.points)
    rotated_points = np.dot(points, rotation_matrix.T)  # Dot product with the rotation matrix

    # Set the new points to the point cloud
    pcd.points = o3d.utility.Vector3dVector(rotated_points)

    return pcd

def get_current_image(client):
    response = client.simGetImage("bottom_center", airsim.ImageType.Scene)
    image = Image.open(io.BytesIO(response))

    # Convert image to numpy array (PIL gives RGB)
    image_array = np.array(image)

    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Normalize pixel values to [0, 1] for gamma correction
    image_array_normalized = image_array / 255.0

    # Gamma correction (exposure reduction)
    exposure_reduction_factor = 1  # >1 darkens the image; increase for more reduction
    image_gamma_corrected = np.power(image_array_normalized, exposure_reduction_factor)

    # Scale back to [0, 255]
    image_gamma_corrected = np.clip(image_gamma_corrected * 255, 0, 255).astype(np.uint8)

    return image_gamma_corrected

def calculate_focal_length_from_fov(image_width_px, image_height_px, fov_deg):
    fov_rad = math.radians(fov_deg)
    fx = (image_width_px / 2) / math.tan(fov_rad / 2)
    fy = (image_height_px / 2) / math.tan(fov_rad / 2)
    return fx, fy

def map_coord_to_pixel(image, points, fx, fy):
    h, w = image.shape[:2]
    cx = w / 2
    cy = h / 2

    point_to_pixel_map = {}

    for point in points:
        x, y, z = point

        # Avoid division by zero (ignore points behind the camera or with z near zero)
        if z <= 0.1:
            continue

        u = int((x * fx) / z + cx)
        v = int((y * fy) / z + cy)

        if 0 <= u < w and 0 <= v < h:
            point_to_pixel_map[(u, v)] = point

    return point_to_pixel_map

def find_closest_voxel_to_pixel(start_pixel, pixel_to_point_map, max_radius=50):
    visited = set()
    queue = deque()
    queue.append(start_pixel)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 4-neighborhood

    while queue:
        u, v = queue.popleft()
        if (u, v) in pixel_to_point_map:
            return pixel_to_point_map[(u, v)]  # Return the 3D point

        visited.add((u, v))

        for dx, dy in directions:
            nu, nv = u + dx, v + dy
            if (nu, nv) not in visited:
                if abs(nu - start_pixel[0]) <= max_radius and abs(nv - start_pixel[1]) <= max_radius:
                    queue.append((nu, nv))

    return None

def land(client):
    distance_sensor_data = client.getDistanceSensorData("Distance", "Drone1")
    distance = distance_sensor_data.distance
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    target_z = pos.z_val + distance - 1 # -1 so that it doesn't go so fast into the ground and stops a bit before it

    print(f"Distance to ground: {distance}, Moving to z: {target_z}") 

    client.moveToZAsync(target_z, 2).join() 
    # client.landAsync().join()
    print("Done Landing")

def get_open3d_point_cloud(lidar_points):
    points = np.array(lidar_points, dtype=np.float32).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd 

def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("Started")

    #Takeoff
    client.takeoffAsync().join()
    client.moveToZAsync(-50, 2).join() 

    # Get Lidar data
    lidar_data = client.getLidarData('GPULidar1', 'Drone1')
    print(lidar_data)
    pcd_raw = get_open3d_point_cloud(lidar_data.point_cloud)
    pcd = rotate_point_cloud(pcd_raw, angle_deg=-90)
    points = np.asarray(pcd.points)

    # Get image data
    image = get_current_image(client)
    image = get_current_image(client)
    image = get_current_image(client)
    image = get_current_image(client)
    image = get_current_image(client)
    image = get_current_image(client)
    image = get_current_image(client)
    image = get_current_image(client)
    image = get_current_image(client)

    # Pick random landing spot (take pixel coord from image find voxel closest to pixel) get out a coordinate point 3d
    h_img, w_img = image.shape[:2]
    landing_center = (int(w_img * 0.40), int(h_img * 0.40))
    fx, fy = calculate_focal_length_from_fov(w_img, h_img, 120)
    coord_to_pixel = map_coord_to_pixel(image, points, fx, fy)
    landing_voxel = find_closest_voxel_to_pixel(landing_center, coord_to_pixel)

    # Find coord relative to real-world
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position

    target_coords = (pos.x_val - landing_voxel[0], pos.y_val + landing_voxel[1])
    client.moveToPositionAsync(target_coords[0], target_coords[1], pos.z_val, velocity=3.0).join()

    land(client)

    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    main()
