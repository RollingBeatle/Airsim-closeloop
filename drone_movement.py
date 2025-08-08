#This combines both monocular and stereo landing pipelines for a drone using AirSim.
# It captures images, overlays a grid, asks an LLM for a target cell, and
# computes the drone's position to land accurately on the target cell.

import cosysairsim as airsim
import numpy as np
import time


class DroneMovement:
    
    def __init__(self, initial_pose = (0,-35,-100)):
        self.client = airsim.MultirotorClient()
        self.initial_pos = initial_pose

    def position_drone(self, fixed=True):
    # Position the drone randomly in demo
        self.client.confirmConnection()
        
        self.client.enableApiControl(True); self.client.armDisarm(True)
        self.client.takeoffAsync().join(); time.sleep(1)
        if fixed:
            x,y,z = self.initial_pos
            self.client.moveToPositionAsync(x,y,z,3).join(); time.sleep(2)
        else:
            z0 = -np.random.uniform(40, 50)
            self.client.moveToZAsync(z0,2).join(); time.sleep(1)

    def move_drone(self, tx, ty, tz):
        self.client.moveToPositionAsync(tx,ty,tz,3).join(); time.sleep(2)
        
        # 5) go down to a desired z
        distz = self.get_rangefinder()
        print("distance to surface", distz)
        targetz = tz + (distz*0.8) 
        print("target position", targetz)
        self.client.moveToZAsync(targetz,3).join()
        return self.get_rangefinder()
        

    def get_rangefinder(self):

        distance_sensor_data = self.client.getDistanceSensorData("Distance","airsimvehicle")
        distance = distance_sensor_data.distance
        print(f"Distance to ground: {distance:.2f}")
        return distance
    
    def land_drone(self): 

        pose = self.client.getMultirotorState().kinematics_estimated.position
        self.client.landAsync().join()
        time.sleep(5)
        landing_dist = self.get_rangefinder()
        landing_dist = pose.z_val + landing_dist - 2
        self.client.moveToZAsync(landing_dist,3).join()
        # self.client.armDisarm(False)


