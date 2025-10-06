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

## Testing 
As a testing environment we use the Unreal Environment *City Sample* with a few changes in the rooftops of some of the tested buildings, originally the rooftops or both buildings are empty, for our testing purposes we added some obstructions like vents or rubble that are available through the map. Our current testing sites are:
![Scenario 1](LVLMRecovery/src/samples/general/ground_truth_scenario_1.jpg)


![Scenario 2](LVLMRecovery/src/general/ground_truth_scenario_2.jpg)

- As the coordinate system is dependent on the site the drone spawn the object `PlayerStart` has to be moved depending the scenario
- `PlayerStart` coordinates for Scenario 1: `(X=69789.929909,Y=-14527.701608,Z=158.775536)`
- `PlayerStart` coordinates for Scenario 2: `(X=36226.923474,Y=-24778.931072,Z=181.969621)`