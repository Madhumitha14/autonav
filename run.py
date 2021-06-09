from environment import CarlaEnvironment
import time

env = CarlaEnvironment()
env.reset()
env.add_rgb_camera()
time.sleep(10)
env.cleanup()
