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

actor_list = []
try:
    client = carla.Client('localhost', 2000)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('audi')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.1))
    actor_list.append(vehicle)
    time.sleep(50)


finally:
    for item in actor_list:
        item.destroy()
    print('done')
