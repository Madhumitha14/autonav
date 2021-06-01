import time
import random
import sys
import os
import glob
import logging

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
        self.synchronous_master = synchronous_master

    def spawn_npc(self, number_of_vehicles=100):
        vehicles_list = []
        walkers_list = []
        all_id = []
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
        print(vehicle_blueprints)
        for n, transform in enumerate(self.spawn_points):
            print('how many times')
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

        for response in self.client.apply_batch_sync(batch, self.synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # SPAWN WALKERS
        percentagePedestriansRunning = 0.0
        percentagePedestriansCrossing = 0.0
        spawn_points = []
        for i in range(number_of_vehicles):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(walker_blueprints)
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    walker_speed.append(walker_bp.get_attribute(
                        'speed').recommended_values[1])
                else:
                    walker_speed.append(walker_bp.get_attribute(
                        'speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp,
                         carla.Transform(), walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)
        self.world.tick()
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            all_actors[i].start()
            all_actors[i].go_to_location(
                self.world.get_random_location_from_navigation())
            all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))
        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' %
              (len(vehicles_list), len(walkers_list)))
        traffic_manager.global_percentage_speed_difference(30.0)

        start_time = time.time()

        while True:
            self.world.tick()

            if int(time.time() - start_time) > 60:
                break

        self.cleanup(vehicles_list, walkers_list, all_actors)

    def cleanup(self, vehicles_list, walkers_list, all_actors):
        pass
        print('\ndestroying %d vehicles' % len(vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x)
                                 for x in vehicles_list])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()
        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
        time.sleep(0.5)


env1 = CarlaEnvironment()
env1.spawn_npc()
