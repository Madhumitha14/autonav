import time
import random
import sys
import os
import glob

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
# END OF IMPORT CARLA

# CONSTANTS
SHOW_PREVIEW = False


class CarlaEnvironment:

    WALKER_LIST = []

    def __init__(self, map='Town01'):
        self.client = carla.Client('localhost', 2000)
        # self.client.set_timeout(2.0)
        self.world = self.client.load_world(map)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()

    def spawn_npc(self, number_of_vehicles=100):
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        vehicle_blueprints = self.blueprint_library.filter('vehicle.*')
        vehicle_blueprints = sorted(vehicle_blueprints, key=lambda bp: bp.id)
        walker_blueprints = self.blueprint_library.filter(
            'walker.pedestrian.*')

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor


env1 = CarlaEnvironment()
