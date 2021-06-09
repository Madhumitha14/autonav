import time
import numpy as np
from collections import deque
from keras.applications.xception import Xception
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from modified_tensorboard import ModifiedTensorBoard
import tensorflow as tf
from threading import Thread
from tqdm import tqdm
backend = tf.compat.v1.keras.backend

MINIMUM_REPLAY_MEMORY_SIZE = 1_000
REPLAY_MEMORY_SIZE = 50_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
UPDATE_TARGET_EVERY = 5

DISCOUNT = 0.99


class DQNAgent:
    def __init__(self, im_height, im_width):
        self.im_height = im_height
        self.im_width = im_width
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/Xception-{int(time.time())}")  # noqa
        self.target_update_counter = 0
        self.graph = tf.Graph()
        self.terminate = False
        self.last_logged_episode = 0
        self.training_initialized = False

    def create_model(self):
        base_model = Xception(weights=None, include_top=False, input_shape=(self.im_height, self.im_width, 3))  # noqa
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(7, activation="linear")(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(loss="mse", optimizer='adam', metrics=['accuracy'])  # noqa
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self):
        if len(self.replay_memory) < MINIMUM_REPLAY_MEMORY_SIZE:
            return
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_states = np.array([transition[0] for transition in minibatch])/255  # noqa
        with tf.GradientTape() as tape:
            current_qs_list = self.model.predict(current_states, PREDICTION_BATCH_SIZE)  # noqa
        new_current_states = np.array([transition[3] for transition in minibatch])/255  # noqa
        with tf.GradientTape() as tape:
            future_qs_list = self.target_model.predict(new_current_states, PREDICTION_BATCH_SIZE)  # noqa

        X = []
        y = []

        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward
            current_qs = current_qs_list[index]
            current_qs[action] = new_q
            X.append(current_state)
            y.append(current_qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_logged_episode = self.tensorboard.step

        with tf.GradientTape() as tape:
            self.model.fit(np.array(X)/255, np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0, shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)  # noqa

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape)/255)[0]  # noqa

    def set_training_initialized(self, value):
        self.training_initialized = True        

    def get_training_initialized(self):
        return self.training_initialized

    def train_in_loop(self):
        X = np.random.uniform(size=(1, self.im_height, self.im_width, 3)).astype(np.float32)  # noqa
        y = np.random.uniform(size=(1, 7)).astype(np.float32)
        with tf.GradientTape() as tape:
            print(" WOOOOOOOOOOOOOOOOOOOHOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
            self.model.fit(X, y, verbose=False, batch_size=1)
        self.set_training_initialized(True)

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.01)
