import cosysairsim as airsim
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import os
import io
from PIL import Image, ImageEnhance


# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()

# Get Lidar data
lidar_data = client.getLidarData('GPULidar1', 'Drone1')
print(lidar_data)

# Your flat array (replace with your data)
points = np.array(lidar_data.point_cloud)  # Convert list to NumPy array
points = points.reshape(-1, 3)

# Compute the distance of each point from the origin (0, 0, 0)
distances = np.linalg.norm(points, axis=1)

# Normalize the distance to range [0, 1] for color mapping
distance_min, distance_max = distances.min(), distances.max()
normalized_distances = (distances - distance_min) / (distance_max - distance_min)

# Map the normalized distance to a color (brighter for closer, dimmer for farther)
# We can use the 'jet' colormap for a nice range from blue (far) to red (close)
colors = plt.cm.jet(normalized_distances)[:, :3]  # Get RGB values from colormap

# Create the Open3D point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.io.write_point_cloud("point_cloud_data.pcd", pcd)
# with open("point_cloud_data.txt", "w") as f:
#     for point in points:
#         f.write(f"{point[0]} {point[1]} {point[2]}\n")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

response = client.simGetImage("bottom_center", airsim.ImageType.Scene)

# Check if image was retrieved successfully
if response is None:
    print("Failed to get image from AirSim")
else:
    # Convert the image bytes into a PIL image
    image = Image.open(io.BytesIO(response))

    # Convert image to numpy array
    image_array = np.array(image)

    # Normalize the pixel values to [0, 1] for exposure adjustment
    image_array_normalized = image_array / 255.0

    # Reduce exposure using gamma correction (non-linear transformation)
    exposure_reduction_factor = 8.5  # Adjust this to control the exposure reduction (smaller is more reduction)
    image_array_exposure_reduced = np.power(image_array_normalized, exposure_reduction_factor)

    # Convert back to [0, 255] and ensure the values stay within valid range
    image_array_exposure_reduced = np.clip(image_array_exposure_reduced * 255, 0, 255).astype(np.uint8)

    # Convert numpy array back to PIL image
    image_exposure_reduced = Image.fromarray(image_array_exposure_reduced)

    # Save the modified image
    image_exposure_reduced.save('image_wide.png')

    print("Image saved with reduced exposure.")

# # get numpy array
# img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) 

# # reshape array to 4 channel image array H X W X 4
# img_rgb = img1d.reshape(response.height, response.width, 3)

# # original image is fliped vertically
# img_rgb = np.flipud(img_rgb)

# # write to png 
# airsim.write_png(os.path.normpath('image_wide.png'), img_rgb)
