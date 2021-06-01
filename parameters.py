import glob
import os
import sys

CARLA_PATH = 'C:/Users/saile/Desktop/Sailesh/Carla Simulator/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg'

try:
    sys.path.append(glob.glob(CARLA_PATH % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import random
import time

class CurrentSimulation:
    def __init__(self):
        try:
            self.carla = carla
            self.client = carla.Client('localhost', 2000)
            self.world = client.get_world()

        finally:
            print('done')

    def get_world():
        print(self.carla.world)
