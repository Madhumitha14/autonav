import os
import time
from threading import Thread
import numpy as np
from tqdm import tqdm
import math

from environment import CarlaEnvironment
from deep_q_learning import DQNAgent

IM_HEIGHT = 300
IM_WIDTH = 533
EPISODES = 100
AGGREGATE_STATS_EVERY = 10
MIN_REWARD = -200
FPS = 20

epsilon = 1
EPSILON_DECAY = 0.95
MIN_EPSILON = 0.001

episode_reward_tracker = [-200]

agent = DQNAgent(IM_HEIGHT, IM_WIDTH)

env = CarlaEnvironment(IM_HEIGHT, IM_WIDTH)
env.reset()
env.add_rgb_camera()
env.add_collision_sensor()
env.add_lane_invasion()

trainer_thread = Thread(target=agent.train_in_loop, daemon=True)
trainer_thread.start()

while True:
    if(agent.get_training_initialized()):
        break
    print('waiting on trainer thread***************************************')
    time.sleep(0.01)

print('trainer thread wating compelted !!!!!!!! ***************************************')

agent.get_qs(np.ones((IM_HEIGHT, IM_WIDTH, 3)))

for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="episodes"):
    env.collisions = []
    agent.tensorboard.step = episode
    episode_reward = 0
    step = 1
    current_state = env.reset()
    done = False
    episode_start = time.time()

    while True:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, 3)
            time.sleep(1/FPS)
        new_state, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.update_replay_memory((current_state, action, reward, new_state, done))  # noqa
        current_state = new_state
        step += 1
        print('running')

        if done:
            print('done')
            break

    env.cleanup()
    episode_reward_tracker.append(episode_reward)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(episode_reward_tracker[-AGGREGATE_STATS_EVERY:]) / len(episode_reward_tracker[-AGGREGATE_STATS_EVERY:])  # noqa
        min_reward = min(episode_reward_tracker[-AGGREGATE_STATS_EVERY:])
        max_reward = max(episode_reward_tracker[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(average_reward=average_reward, min_reward=min_reward, max_reward=max_reward, epsilon=epsilon)  # noqa

        if min_reward > MIN_REWARD:
            agent.model.save(f"models/Xception_{max_reward:_>7.2f}max_{min_reward:_>7.2f}avg_{average_reward:_>7.2f}min_{int(time.time())}")  # noqa

    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

agent.terminate = True
trainer_thread.join()
agent.model.save(f'models/Xception_{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')  # noqa
