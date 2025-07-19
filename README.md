# Autonomous Drone Recovery
This is the **prototype** for our ADR system based on the Airsim-CoSyS simulation environment. 
## Structure
- `airsim_settings` folder contains different configurations of the drones
- `Tracking` includes an implementation of object tracking to position the drone on top of a desired point
- `LiDAR` includes the data gathering tools to get the flat surfaces based on LiDAR
- `depth_estimation` contains basic implementations of monocular and stereo depth estimation

## Running the loop
The main file to run is ```close_loop.py```, this will create auxiliary directories that will be used through the process, while `drone_movement.py` is where the drone performs most physical actions and does the image processing.
> Note: this structure will change to properly decouple the processing with the movement
