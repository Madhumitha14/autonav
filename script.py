import time
import random
import sys
import os
import glob
import logging
import numpy as np
import cv2

# IMPORT CARLA
CARLA_PATH = 'C:/Users/saile/Desktop/Sailesh/Carla Simulator/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg'

try:
    sys.path.append(glob.glob(CARLA_PATH % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla  # noqa
from carla import VehicleLightState as vls  # noqa
# END OF IMPORT CARLA

# CONSTANTS
SHOW_PREVIEW = False


class CarlaEnvironment:

    def __init__(self, map='Town01', synchronous_master=True):
        self.client = carla.Client('localhost', 2000)
        # self.client.set_timeout(2.0)
        self.world = self.client.load_world(map)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.map.get_spawn_points()

    # def process_img(self, image):
    #     i = np.array(image.raw_data)
    #     # print(i.shape)
    #     i2 = i.reshape((self.im_height, self.im_width, 4))
    #     i3 = i2[:, :, :3]
    #     if self.SHOW_CAM:
    #         cv2.imshow("", i3)
    #         cv2.waitKey(1)
    #     self.front_camera = i3
    #     return i3/255.0

    # def step(self, action):
    #     if action == 0:
    #         self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMOUNT))
    #         pass
    #     elif action == 1:
    #         pass
    #     elif action == 2:
    #         pass
