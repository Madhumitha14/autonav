from environment import CarlaEnvironment
from collections import deque
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.model import Model
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from threading import Thread
from tqdm import tqdm


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=5000)
        self.target_update_counter = 0
