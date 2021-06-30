import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import os
import utils

INPUT_SHAPE = (66, 200, 3)
DATA_DIR = "C:\\Users\\saile\\Desktop\\Train"


def load_data():
    data_df = pd.read_csv(os.path.join(DATA_DIR, "driving_log.csv"), names=[
                          "center", "left", "right", "steering", "throttle", "reverse", "speed"])
    X = data_df[["center", "left", "right"]].values
    y = data_df["steering"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8)
    return X_train, X_test, y_train, y_test


def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation="elu", strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), activation="elu", strides=(1, 1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))
    model.summary()
    return model


def train_model(model, X_train, X_test, y_train, y_test):
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor="val_loss",
                                 verbose=0,
                                 save_best_only=True,
                                 mode="auto")
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit_generator(utils.batch_generator(DATA_DIR, X_train, y_train, 40, False),
                                            20000, 10, max_queue_size=1,
                                            validation_data=utils.batch_generator(DATA_DIR, X_test, y_test, 40, False),  # noqa
                                            validation_steps=len(X_test), callbacks=[checkpoint], verbose=1)


def main():
    data = load_data()
    model = build_model()
    train_model(model, *data)


if __name__ == "__main__":
    main()
