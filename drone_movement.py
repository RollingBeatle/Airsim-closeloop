#This combines both monocular and stereo landing pipelines for a drone using AirSim.
# It captures images, overlays a grid, asks an LLM for a target cell, and
# computes the drone's position to land accurately on the target cell.

import cosysairsim as airsim
import numpy as np
import time
import math
from pynput import keyboard
import cv2


class DroneMovement:
    
    def __init__(self, initial_pose = (0,-35,-100), debug=True):
        self.client = airsim.MultirotorClient()
        self.initial_pos = initial_pose
        self.client.confirmConnection()
        
        self.client.enableApiControl(True); self.client.armDisarm(True)
        self.client.takeoffAsync().join(); time.sleep(1)

        self.speed = 5  # m/s
        self.duration = 0.5 
        self.debug = debug
    
    def position_drone(self, fixed=True, position=(0,0,0), ori=None):
    # Position the drone randomly in demo
        if fixed:
            x,y,z = self.initial_pos
            self.client.moveToPositionAsync(x,y,z,3).join()
        
        else:
            z0 = -np.random.uniform(230,250)
            pose = self.client.getMultirotorState().kinematics_estimated.position
            self.client.moveToZAsync(pose.z_val+z0, 2).join(); time.sleep(2)
            x,y,z = position
            
            self.client.moveToPositionAsync(x,y,z0,3).join(); time.sleep(5)
            self.client.moveToPositionAsync(x,y,z,3).join(); time.sleep(5)
            if ori:
                pos = self.client.getMultirotorState().kinematics_estimated.position
                pose = airsim.Pose(pos, ori)
                self.client.simSetVehiclePose(pose,True)
                time.sleep(2)
        

    def move_drone(self, tx, ty, tz):

        self.client.moveToPositionAsync(tx,ty,tz,3).join(); time.sleep(2)
        
        return self.move_to_z(tz)
    
    def move_to_z(self, tz):

        distz = self.get_rangefinder()
        print("distance to surface", distz)
        targetz = tz + (distz*0.8) 
        print("target position", targetz)
        self.client.moveToZAsync(targetz,3).join();time.sleep(2)
        return self.get_rangefinder()
        

    def get_rangefinder(self):

        distance_sensor_data = self.client.getDistanceSensorData("Distance","airsimvehicle")
        distance = distance_sensor_data.distance
        print(f"Distance to ground: {distance:.2f}")
        return distance
    
    def land_drone(self): 

        pose = self.client.getMultirotorState().kinematics_estimated.position
        
        landing_dist = self.get_rangefinder()
        landing_dist = pose.z_val + landing_dist - 2
        print("Landing drone")
        self.client.moveToZAsync(landing_dist,3).join()
        self.client.landAsync().join()
        time.sleep(5)
        # self.client.armDisarm(False)
    

    def manual_control(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()
        self.running = True
        self.command_queue = []
        print("Controls: w/s = forward/back | a/d = left/right | q/e = up/down | x = exit | k = position| p = take pictures")
        while self.running:
            if self.command_queue:
                k = self.command_queue.pop(0)
                if self.debug: print(f"pressed {k}")
                if k == 'w':
                    self.client.moveByVelocityAsync(self.speed, 0, 0, self.duration).join()
                elif k == 's':
                    self.client.moveByVelocityAsync(-self.speed, 0, 0, self.duration).join()
                elif k == 'a':
                    self.client.moveByVelocityAsync(0, -self.speed, 0, self.duration).join()
                elif k == 'd':
                    self.client.moveByVelocityAsync(0,self.speed, 0, self.duration).join()
                elif k == 'q':
                    self.client.moveByVelocityAsync(0, 0, -self.speed, self.duration).join()
                elif k == 'e':
                    self.client.moveByVelocityAsync(0, 0, self.speed, self.duration).join()
                elif k == 'm':
                    self.client.moveByAngleThrottleAsync(0,0,self.speed,-self.speed,self.duration).join()
                elif k == 'n':
                    self.client.moveByAngleThrottleAsync(0,0,self.speed,self.speed,self.duration).join()
                elif k == 'k':
                    print("Estimated position",self.client.getMultirotorState().kinematics_estimated.position)
                    print("Estimated angle", self.client.getMultirotorState().kinematics_estimated.orientation)
                elif k == "p":
                    # TODO: set cam name with a variable
                    resp = self.client.simGetImages([airsim.ImageRequest("frontcamera",airsim.ImageType.Scene,False,False)])[0]
                    img = np.frombuffer(resp.image_data_uint8, np.uint8).reshape(resp.height,resp.width,3)
                    # save image
                    cv2.imwrite("images/manual-pic.jpg", cv2.cvtColor(img,cv2.COLOR_RGB2BGR))

    def on_press(self, key):
        keys = ['w', 's', 'a', 'd', 'q', 'e','k','m', 'n', 'p']
        try:
            k = key.char.lower()
            if k == 'x':
                self.running = False
            elif k in keys:
                self.command_queue.append(k)
        except AttributeError:
            pass
    
    def distance_2d(self, pos1, pos2):
        return math.sqrt((pos1.x_val - pos2[0])**2 + (pos1.y_val - pos2[1])**2)
        

    



