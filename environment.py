import time
import random
import sys
import os
import glob
import numpy as np
import cv2
import math

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
IMAGE_SIZE_X = 533
IMAGE_SIZE_Y = 300
SECONDS_PER_EPISODE = 10


class CarlaEnvironment:

    def __init__(self, im_height, im_width, map='Town03'):
        self.client = carla.Client('localhost', 2000)
        self.world = self.client.load_world(map)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.map.get_spawn_points()
        self.im_height = im_height
        self.im_width = im_width

    def reset(self):
        self.actor_list = []
        self.collisions = []
        self.lane_invasions = []
        vehicle_blueprint = self.blueprint_library.filter(f"vehicle.{VEHICLE_NAME}.{MODEL_NAME}")[0]  # noqa
        vehicle_spawn_location = random.choice(self.spawn_points)
        while True:
            self.vehicle = self.world.try_spawn_actor(vehicle_blueprint, vehicle_spawn_location)  # noqa
            if self.vehicle != None:
                break
        self.actor_list.append(self.vehicle)
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))  # noqa
        self.rgb_camera = None
        self.depth_sensor = None
        self.episode_start = time.time()
        return self.rgb_camera

    def add_rgb_camera(self, x=3, y=0, z=2):
        rgb_camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        rgb_camera_bp.set_attribute("image_size_x", f"{self.im_width}")
        rgb_camera_bp.set_attribute("image_size_y", f"{self.im_height}")
        rgb_camera_bp.set_attribute("fov", "120")
        rgb_camera_transform = carla.Transform(carla.Location(x, y, z))
        self.rgb_camera = self.world.try_spawn_actor(rgb_camera_bp, rgb_camera_transform, attach_to=self.vehicle)  # noqa
        self.actor_list.append(self.rgb_camera)
        self.rgb_camera.listen(lambda image: self.process_image(image, 'rgb'))

    def add_depth_sensor(self, x=3, y=0, z=2):
        depth_sensor_bp = self.blueprint_library.find('sensor.camera.depth')
        depth_sensor_bp.set_attribute("image_size_x", f"{self.im_width}")
        depth_sensor_bp.set_attribute("image_size_y", f"{self.im_height}")
        depth_sensor_bp.set_attribute("fov", "120")
        depth_sensor_transform = carla.Transform(carla.Location(x, y, z))
        self.depth_sensor = self.world.try_spawn_actor(depth_sensor_bp, depth_sensor_transform, attach_to=self.vehicle)  # noqa
        self.actor_list.append(self.rgb_camera)
        self.depth_sensor.listen(lambda image: self.process_image(image, 'depth'))  # noqa

    def add_collision_sensor(self):
        collision_sensor_bp = self.blueprint_library.find('sensor.other.collision')  # noqa
        self.collision_sensor = self.world.try_spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=self.vehicle)  # noqa
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self.handle_collision(event))  # noqa

    def handle_collision(self, event):
        self.collisions.append(event)

    def add_lane_invasion(self):
        lane_invasion_sensor_bp = self.blueprint_library.find('sensor.other.lane_invasion')  # noqa
        self.lane_invasion_sensor = self.world.try_spawn_actor(lane_invasion_sensor_bp, carla.Transform(), attach_to=self.vehicle)  # noqa
        self.actor_list.append(self.lane_invasion_sensor)
        self.lane_invasion_sensor.listen(lambda event: handle_lane_invasion(event))  # noqa

    def handle_lane_invasion(self, event):
        self.lane_invasions.append(event)

    def process_image(self, image, camera):
        img = np.array(image.raw_data)
        img = img.reshape((self.im_height, self.im_width, 4))
        img = img[:, :, :3]
        cv2.imshow("Hood Cam", img)
        cv2.waitKey(1)
        if camera == 'rgb':
            self.rgb_camera_data = img
        if camera == 'depth':
            self.depth_sensor_data = img

    def step(self, action):
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))  # noqa
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.5))  # noqa
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1.0))  # noqa
        if action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-0.5))  # noqa
        if action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1.0))  # noqa
        if action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))  # noqa
        if action == 6:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0))  # noqa

        velocity = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2))

        if len(self.collision) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -2
        if len(self.lane_invasions) != 0:
            reward = -1 * len(self.lane_invasions)

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        return self.front_camera, reward, done, None

    def cleanup(self):
        for actor in self.actor_list:
            actor.destroy()
