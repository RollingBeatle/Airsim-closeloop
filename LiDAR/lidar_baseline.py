import cosysairsim as airsim
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import io
from PIL import Image
import cv2
from collections import deque
import math

class LidarMovement:

    def rotate_point_cloud(self, pcd, angle_deg=-90):
        angle_rad = np.radians(angle_deg)
        rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad), 0],
                                    [np.sin(angle_rad), np.cos(angle_rad), 0],
                                    [0, 0, 1]])
        points = np.asarray(pcd.points)
        rotated_points = np.dot(points, rotation_matrix.T)
        pcd.points = o3d.utility.Vector3dVector(rotated_points)
        return pcd

    def get_current_image(self, client, cam_name):
        response = client.simGetImages([airsim.ImageRequest(cam_name,airsim.ImageType.Scene,False,False)])[0]
        img = np.frombuffer(response.image_data_uint8, np.uint8).reshape(response.height,response.width,3)
        image = Image.fromarray(img)
        #image = Image.open(io.BytesIO(response))
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        return image_array

    def calculate_focal_length_from_fov(self, image_width_px, image_height_px, fov_deg):
        fov_rad = math.radians(fov_deg)
        fx = (image_width_px / 2) / math.tan(fov_rad / 2)
        fy = fx
        return fx, fy

    def map_coord_to_pixel(self, image, points, fx, fy):
        h, w = image.shape[:2]
        cx = w / 2
        cy = h / 2
        pixel_to_point_map = {}

        for point in points:
            x, y, z = point
            if z <= 0.1:
                continue
            u = int((x * fx) / z + cx)
            v = int((y * fy) / z + cy)
            if 0 <= u < w and 0 <= v < h:
                pixel_to_point_map[(u, v)] = point

        return pixel_to_point_map


    def find_closest_voxel_to_pixel(self, start_pixel, pixel_to_point_map, max_radius=50):
        sx, sy = start_pixel
        closest_pixel = None
        closest_point = None
        min_dist = float('inf')

        for dx in range(-max_radius, max_radius + 1):
            for dy in range(-max_radius, max_radius + 1):
                u, v = sx + dx, sy + dy
                if (u, v) in pixel_to_point_map:
                    dist = math.sqrt(dx**2 + dy**2)
                    if dist < min_dist:
                        min_dist = dist
                        closest_pixel = (u, v)
                        closest_point = pixel_to_point_map[(u, v)]

        return closest_pixel, closest_point


    def land(self, client):
        distance_sensor_data = client.getDistanceSensorData("Distance", "Drone1")
        distance = distance_sensor_data.distance
        state = client.getMultirotorState()
        pos = state.kinematics_estimated.position
        target_z = pos.z_val + distance - 1 # -1 so that it doesn't go so fast into the ground and stops a bit before it

        print(f"Distance to ground: {distance}, Moving to z: {target_z}") 

        client.moveToZAsync(target_z, 2).join() 
        # client.landAsync().join()
        print("Done Landing")

    def get_open3d_point_cloud(self, lidar_points):
        points = np.array(lidar_points, dtype=np.float32).reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd 

    def crop_image_around_pixel(self, image, center_pixel, size=100):
        u, v = center_pixel
        half = size // 2
        top = max(v - half, 0)
        bottom = min(v + half, image.shape[0])
        left = max(u - half, 0)
        right = min(u + half, image.shape[1])
        crop = image[top:bottom, left:right]
        cv2.imwrite("images/landing_zone_crop.png", crop)
        return crop

    def colorize_point_cloud_with_image(self, points, image, fx, fy):
        h, w = image.shape[:2]
        cx = w / 2
        cy = h / 2

        colors = []
        valid_points = []

        for point in points:
            x, y, z = point
            if z <= 0.1:
                continue

            u = int((x * fx) / z + cx)
            v = int((y * fy) / z + cy)

            if 0 <= u < w and 0 <= v < h:
                bgr = image[v, u]  # OpenCV uses BGR format
                rgb = [bgr[2] / 255.0, bgr[1] / 255.0, bgr[0] / 255.0]
                colors.append(rgb)
                valid_points.append(point)

        # Build colored Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        return pcd


def main():

    lidar_m = LidarMovement()
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("Started")

    client.takeoffAsync().join()
    client.moveToZAsync(-50, 2).join()

    # Get Lidar + image
    lidar_data = client.getLidarData('GPULidar1', 'Drone1')
    pcd_raw = lidar_m.get_open3d_point_cloud(lidar_data.point_cloud)
    pcd = lidar_m.rotate_point_cloud(pcd_raw)
    points = np.asarray(pcd.points)

    image = lidar_m.get_current_image(client)
    image = lidar_m.get_current_image(client)
    image = lidar_m.get_current_image(client)
    image = lidar_m.get_current_image(client)
    image = lidar_m.get_current_image(client)
    image = lidar_m.get_current_image(client)
    image = lidar_m.get_current_image(client)
    image = lidar_m.get_current_image(client)
    image = lidar_m.get_current_image(client)


    h_img, w_img = image.shape[:2]
    landing_center = (int(w_img * 0.30), int(h_img * 0.30))
    print(f"width: {w_img}")
    fx, fy = lidar_m.calculate_focal_length_from_fov(w_img, h_img, 120)
    pixel_to_coord = lidar_m.map_coord_to_pixel(image, points, fx, fy)
    landing_pixel, landing_voxel = lidar_m.find_closest_voxel_to_pixel(landing_center, pixel_to_coord)
    colored_pcd = lidar_m.colorize_point_cloud_with_image(points, image, fx, fy)
    o3d.visualization.draw_geometries([colored_pcd])

    lidar_m.crop_image_around_pixel(image, landing_center, size=100)

    # Get current GPS
    gps_data = client.getGpsData(gps_name="GPS", vehicle_name="Drone1").gnss.geo_point
    current_gps = (gps_data.latitude, gps_data.longitude, gps_data.altitude)

    if landing_voxel is None:
        print("No valid landing voxel found.")
        return

    # Conversion: 1 meter ≈ 0.000009 deg lat, 0.000011 deg lon (approx.)
    METERS_TO_LAT = 0.000009
    METERS_TO_LON = 0.000011

    x_m, y_m, _ = landing_voxel  # x: right, y: forward in LIDAR

    offset_lat = y_m * METERS_TO_LAT
    offset_lon = -x_m * METERS_TO_LON

    target_lat = current_gps[0] + offset_lat
    target_lon = current_gps[1] + offset_lon

    # Estimate NED displacement from GPS difference (in meters)
    dx = -y_m
    dy = x_m # because forward is negative NED x

    print(f"Desired landing pixel: {landing_center}")
    print(f"Actual landing pixel: {landing_pixel}")
    print(f"Current GPS: {current_gps}")
    print(f"Target GPS: (lat: {target_lat}, lon: {target_lon})")
    print(f"Estimated move offset in NED frame: dx={dx:.2f}m, dy={dy:.2f}m")

    # Move using relative position (NED)
    current_pose = client.getMultirotorState().kinematics_estimated.position
    target_x = current_pose.x_val + dx
    target_y = current_pose.y_val + dy
    target_z = current_pose.z_val  # stay at same height

    client.moveToPositionAsync(target_x, target_y, target_z, velocity=3.0).join()

    lidar_m.land(client)

    final_gps = client.getGpsData(gps_name="GPS", vehicle_name="Drone1").gnss.geo_point
    print(f"Final drone GPS: (lat: {final_gps.latitude}, lon: {final_gps.longitude})")

    client.armDisarm(False)
    client.enableApiControl(False)

if __name__ == "__main__":
    main()
