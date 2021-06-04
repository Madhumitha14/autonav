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
VEHICLE_NAME = 'audi'
MODEL_NAME = 'a2'
IMAGE_SIZE_X = 600
IMAGE_SIZE_Y = 400


class CarlaEnvironment:

    def __init__(self, map='Town01'):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.load_world(map)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.map.get_spawn_points()
        self.actor_list = []
        vehicle_blueprint = self.blueprint_library.filter(
            f"vehicle.{VEHICLE_NAME}.{MODEL_NAME}")
        vehilce_spawn_location = random.choice(self.map.get_spawn_points())
        self.vehicle = world.try_spawn_actor(
            vehicle_blueprint, vehilce_spawn_location)
        self.actor_list.append(self.vehicle)

    def add_rgb_camera(self, x=3, y=0, z=2):
        rgb_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute("image_size_x", f"{IMAGE_SIZE_X}")
        rgb_camera_bp.set_attribute("image_size_y", f"{IMAGE_SIZE_Y}")
        rgb_camera_bp.set_attribute("fov", "120")
        rgb_camera_transform = carla.Transform(carla.Location(x, y, z))
        self.rgb_camera = self.world.spawn_actor(
            rgb_camera_bp, rgb_camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_camera)

    def add_depth_sensor(self, x=3, y=0, z=2):
        depth_sensor_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_sensor_bp.set_attribute("image_size_x", f"{IMAGE_SIZE_X}")
        depth_sensor_bp.set_attribute("image_size_y", f"{IMAGE_SIZE_Y}")
        depth_sensor_bp.set_attribute("fov", "120")
        depth_sensor_transform = carla.Transform(carla.Location(x, y, z))
        self.depth_sensor = self.world.spawn_actor(
            depth_sensor_bp, depth_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.rgb_camera)

    def show_preview(self):
        def process_image(image):
            img = np.array(image.raw_data)
            img = img.reshape((IMAGE_SIZE_Y, IMAGE_SIZE_X, 4))
            img = img[:, :, :3]
            cv2.imshow("RGB Camera", img)
            cv2.waitKey(1)
        self.rgb_camera.listen(lambda image: process_image(image))
