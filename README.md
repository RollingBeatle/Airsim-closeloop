# LVLM Assisted Drone Landing
This is the **prototype** for our ADL system based on the Airsim-CoSyS simulation environment. Our paper based on this work is available 
[here](https://arxiv.org/abs/2510.00167)
## Structure
- `LVLMRecovery` folder contains all class and components of the pipeline
- `airsim_settings` folder contains different configurations of the drones
- `Tracking` includes an implementation of object tracking to position the drone on top of a desired point
- `LiDAR` includes the data gathering tools to get the flat surfaces based on LiDAR
- `depth_estimation` contains basic implementations of monocular and stereo depth estimation

## Running the Assisted Drone Landing
- **Prerequisite:** Have a compatible Unreal Engine 5 project with the [Cosys-airsim](https://cosys-lab.github.io/Cosys-AirSim/) plugin installed.
- **Installation:** 

`git clone https://github.com/RollingBeatle/Airsim-closeloop`

`cd Airsim-closeloop`

`python -m pip install -e .`
- Once you have the simulation running you need to create instances of `GPTAgent`, `ImageProcessing` and `DroneMovement`
- Feed the instances to the `main_pipeline` function from `recovery_pipeline` and add parameters specifying LVLM model, drone position, drone orientation, how many times 
to run the application, and a data recording flag

## Testing 
As a testing environment we use the Unreal Environment *City Sample* with a few changes in the rooftops of some of the tested buildings. In closeloop_testing we provide the code to execute our testing in the individual modules as well as the full pipeline in two controlled scenarios. The buildings in our scenarios have been modified to add rubble and other obstacles to make the distinction between safe and unsafe clear for our LVLM. Our current testing sites are:
![Scenario 1](src/LVLMRecovery/samples/ground_truth_scenario1.jpg)

![Scenario 2](src/LVLMRecovery/samples/ground_truth_scenario2.jpg)

- As the coordinate system is dependent on the site the drone spawn the object `PlayerStart` has to be moved depending the scenario
- `PlayerStart` coordinates for Scenario 1: `(X=69789.929909,Y=-14527.701608,Z=158.775536)`
- `PlayerStart` coordinates for Scenario 2: `(X=36226.923474,Y=-24778.931072,Z=181.969621)`