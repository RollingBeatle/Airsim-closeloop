import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import cv2
import math
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors

def load_point_cloud(pcd_file):
    # Load the point cloud from a PCD file
    pcd = o3d.io.read_point_cloud(pcd_file)
    return pcd

def find_flat_areas(points, area_size=2.0, height_tolerance=0.1):
    # Create a KD-tree for efficient spatial querying (using X and Y for flat area detection)
    kdtree = cKDTree(points[:, [0, 1]])  # Only consider X and Y (forward/backward, left/right)

    flat_areas = []

    # Iterate over each point in the cloud
    for i, point in enumerate(points):
        # Query points within the given area size in X and Y (forward/back, left/right)
        neighbors_idx = kdtree.query_ball_point(point[0:2], area_size / 2.0)  # Radius = half of area_size
        neighbors = points[neighbors_idx]
        
        # Extract the neighbors' Z-values (heights)
        z_values = neighbors[:, 2]
        
        # Check if the variation in Z-values (heights) is below the tolerance (flat area)
        if np.max(z_values) - np.min(z_values) < height_tolerance:
            flat_areas.append(point)
    
    return np.array(flat_areas)

def custom_distance(p1, p2):
    # Distance in X, Y, Z directions
    xy_distance = np.linalg.norm(p1[:2] - p2[:2])  # Euclidean distance in x and y
    z_distance = abs(p1[2] - p2[2])  # Difference in z

from sklearn.cluster import DBSCAN
import numpy as np

def cluster_landing_zones(flat_areas, distance_threshold=3, min_cluster_size=350, z_threshold=0.5):
    """
    Cluster flat areas into larger landing zones. 
    The distance threshold determines how close points need to be to be clustered together in x and y.
    The z_threshold ensures that points are clustered together only if their z difference is within the threshold.
    The min_cluster_size ensures that only large enough clusters are considered.
    """
    # Scale z-axis for clustering
    scaled_flat_areas = np.copy(flat_areas)
    scaled_flat_areas[:, 2] *= distance_threshold / z_threshold  # Scale z by the ratio of distance_threshold to z_threshold

    # Step 1: Use DBSCAN for local clustering
    db = DBSCAN(eps=distance_threshold, min_samples=min_cluster_size).fit(scaled_flat_areas)

    # Extract the labels (cluster IDs)
    labels = db.labels_

    # Group points into clusters based on DBSCAN labels
    clusters = []
    for label in set(labels):
        if label == -1:
            continue  # Skip noise points
        
        # Extract points belonging to this cluster
        cluster_points = flat_areas[labels == label]
        
        # Check if the cluster size is larger than a threshold
        if len(cluster_points) >= min_cluster_size:
            clusters.append(cluster_points)

    return clusters



def extract_image_for_landing_zone(image, landing_zone, fx, fy, cx, cy, padding=20):
    # Get bounding box of the landing zone in 2D image space
    min_u, min_v = float('inf'), float('inf')
    max_u, max_v = float('-inf'), float('-inf')

    for x, y, z in landing_zone:
        # Convert the 3D point to 2D image coordinates (u, v)
        u = int((fx * x / z) + cx)
        v = int((fy * y / z) + cy)
        min_u = min(min_u, u)
        min_v = min(min_v, v)
        max_u = max(max_u, u)
        max_v = max(max_v, v)

    # Add padding to the bounding box for more context
    min_u = max(min_u - padding, 0)
    min_v = max(min_v - padding, 0)
    max_u = min(max_u + padding, image.shape[1] - 1)  # Image width
    max_v = min(max_v + padding, image.shape[0] - 1)  # Image height

    # Calculate dynamic resolution based on the size of the bounding box
    width = abs(max_u - min_u)
    height = abs(max_v - min_v)
    print("The width, height: ", width, height)
    # Set minimum resolution thresholds
    min_resolution = 100  # Minimum width/height resolution
    scale_factor = max(min_resolution / min(width, height), 1.0)  # Ensure minimum resolution
    
    # Resize the extracted image dynamically
    extracted_image = image[min_v:max_v, min_u:max_u]
    new_width = int(extracted_image.shape[1] * scale_factor)
    new_height = int(extracted_image.shape[0] * scale_factor)
    scaled_min_u = int(min_u * scale_factor)
    scaled_max_u = int(max_u * scale_factor)
    scaled_min_v = int(min_v * scale_factor)
    scaled_max_v = int(max_v * scale_factor)
    resized_image = cv2.resize(extracted_image, (new_width, new_height))

    return resized_image, (scaled_min_v, scaled_min_u, scaled_max_v, scaled_max_u)

def visualize_point_cloud_with_landing_zones(points, landing_zones):
    # Set up colors for the point cloud
    colors = np.zeros((points.shape[0], 3))
    colors[:, 2] = 1  # Set all points to blue (0, 0, 1)
    
    # Set landing zones to green
    for zone in landing_zones:
        for point in zone: 
            distances = np.linalg.norm(points - point, axis=1)
            closest_point_idx = np.argmin(distances)
            colors[closest_point_idx] = [0, 1, 0]  # Green for landing zone points

    # Create Open3D point cloud object with updated colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

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

def apply_segmentation_to_point_cloud(pcd, segmentation_image, fx, fy, cx, cy):
    # Rotate the point cloud by -90 degrees before applying segmentation
    pcd = rotate_point_cloud(pcd, angle_deg=-90)
    
    points = np.asarray(pcd.points)

    # Initialize colors for the point cloud
    colors = np.zeros((points.shape[0], 3))

    # Loop through each point in the point cloud
    for i, (x, y, z) in enumerate(points):
        # Check if z is valid (non-zero and non-NaN)
        if z != 0 and not np.isnan(z):
            try:
                # Calculate the 2D image coordinates (u, v) corresponding to the 3D point (x, y, z)
                u = int((fx * x / z) + cx)  # Projection onto the image's horizontal axis
                v = int((fx * y / z) + cy)  # Projection onto the image's vertical axis

                # Check if the (u, v) coordinates fall within the image bounds
                if 0 <= u < segmentation_image.shape[1] and 0 <= v < segmentation_image.shape[0]:
                    # Get the segmentation color at (v, u)
                    color = segmentation_image[v, u, :]  # Assuming RGB format
                    colors[i] = color / 255.0  # Normalize to [0, 1] for Open3D

            except Exception as e:
                print(f"Error processing point {i}: {e}")
                continue
        else:
            print(f"Skipping point {i} due to invalid z value.")

    # Assign the colors to the point cloud
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Save or visualize the updated point cloud
    o3d.visualization.draw_geometries([pcd])

def calculate_focal_length_from_fov(image_width_px, image_height_px, fov_deg):
    fov_rad = math.radians(fov_deg)
    fx = (image_width_px / 2) / math.tan(fov_rad / 2)
    fy = (image_height_px / 2) / math.tan(fov_rad / 2)
    return fx, fy

def find_roofs(pcd_file, seg_img):
    # Load the point cloud from the PCD file
    pcd_file = pcd_file
    pcd = load_point_cloud(f"point_cloud_data/{pcd_file}")
    pcd  = rotate_point_cloud(pcd, angle_deg=-90)
    
    # Get the points from the PCD
    points = np.asarray(pcd.points)

    # Adjust the area size and tolerance as needed
    area_size = 2
    height_tolerance = 0.1

    # Find flat areas
    flat_areas = find_flat_areas(points, area_size=area_size, height_tolerance=height_tolerance)

    # Cluster landing zones with stricter distance and size criteria
    landing_zones = cluster_landing_zones(flat_areas, distance_threshold=4.0, min_cluster_size=250, z_threshold=0.2)

    # Visualize landing zones
    # visualize_point_cloud_with_landing_zones(points, landing_zones)

    # Calculate focal lengths
    fx, fy = calculate_focal_length_from_fov(960, 540, 90)

    # Load the segmentation image
    segmentation_image = cv2.imread(f"images/{seg_img}")
    # center of the image
    cx, cy = 960//2,540//2 
    areas = []
    # Extract images for each landing zone
    for i, zone in enumerate(landing_zones):
            try:
                extracted_image, area = extract_image_for_landing_zone(segmentation_image, zone, fx, fy, cx, cy)
                
                areas.append((area,extracted_image))
                cv2.imwrite(f"landing_zones/landing_zone_{i}.png", extracted_image)
            except:
                continue
    return areas

if __name__ == "__main__":
    find_roofs("point_cloud_1.pcd", "img_1.png")
