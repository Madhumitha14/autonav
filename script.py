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
from carla import VehicleLightState as vls  # noqa
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
        self.spawn_points = self.map.get_spawn_points()

    def spawn_npc(self, number_of_vehicles=100):
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        vehicle_blueprints = self.blueprint_library.filter('vehicle.*')
        vehicle_blueprints = sorted(vehicle_blueprints, key=lambda bp: bp.id)
        walker_blueprints = self.blueprint_library.filter(
            'walker.pedestrian.*')

        # SPAWN VEHICLES
        batch = []
        if len(self.spawn_points) < number_of_vehicles:
            number_of_vehicles = self.spawn_points
        random.shuffle(self.spawn_points)
        for n, transform in enumerate(self.spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(vehicle_blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(
                    blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(
                    blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        SetVehicleLightState = carla.command.SetVehicleLightState
        FutureActor = carla.command.FutureActor

        # prepare the light state of the cars to spawn
        light_state = vls.Position | vls.LowBeam | vls.LowBeam

        # spawn the cars and set their autopilot and light state all together
        batch.append(SpawnActor(blueprint, transform)
                     .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
                     .then(SetVehicleLightState(FutureActor, light_state)))

        self.world.tick()
        time.sleep(60)


env1 = CarlaEnvironment()
env1.spawn_npc()
