import time
import random
import glob
import os
import sys

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


class AvailableElements:
    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()

    def get_client_info(self):
        return {
            client_version: self.client.get_client_version(),
            server_version: self.client.get_server_version(),
            world: self.world,
            summary: self.world.__str__()
        }

    def get_available_maps(self):
        return self.client.get_available_maps()

    def get_blueprints(self):
        blueprints = [
            bp for bp in self.world.get_blueprint_library().filter('*')]
        for blueprint in blueprints:
            print(blueprint.id)
            for attr in blueprint:
                print('  - {}'.format(attr))


current_world = AvailableElements()
current_world.get_blueprints()
