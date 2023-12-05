import multiprocessing
import os
from multiprocessing import Pool

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential
from keras.src.optimizers import Adam
from keras.src.losses import categorical_crossentropy
from keras.src.utils import to_categorical
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

PADDING = 0.5
RESOLUTION = 0.01
HIDDEN_LAYERS = [(2, "sigmoid")]

PATIENCE = 32

iris_data = load_iris()

x = iris_data.data[:, 2:]
encoder = LabelEncoder()
y = encoder.fit_transform(iris_data.target)


def get_model(_hidden_layers):
    new_model = Sequential()
    for hidden_layer_shape, hidden_layer_activation in _hidden_layers:
        new_model.add(Dense(hidden_layer_shape, activation=hidden_layer_activation))
    new_model.add(Dense(3, activation='softmax'))
    new_model.build([1, 2])
    return new_model


def process_image(_epoch):
    root_path = os.path.join(*[".", str(HIDDEN_LAYERS)])
    image_path = os.path.join(*[root_path, "images", str(_epoch + 1) + ".png"])
    print("Processing " + image_path)

    xa, xb = np.meshgrid(np.arange(start=x[:, 0].min() - PADDING, stop=x[:, 0].max() + PADDING, step=RESOLUTION),
                         np.arange(start=x[:, 1].min() - PADDING, stop=x[:, 1].max() + PADDING, step=RESOLUTION))

    model = get_model(HIDDEN_LAYERS)
    model.load_weights(os.path.join(*[root_path, "weights", str(_epoch + 1) + ".weights.h5"]))
    model.compile(optimizer=Adam(), loss=categorical_crossentropy)

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.contourf(xa, xb,
                np.argmax(model.predict(np.array([xa.ravel(), xb.ravel()]).T, verbose=0), axis=1).reshape(
                    xa.shape),
                cmap=ListedColormap(('red', 'green', 'blue')), alpha=0.2)

    for i, j in enumerate(np.unique(y)):
        ax.scatter(x[y == j, 0], x[y == j, 1], s=2.5, color=ListedColormap(('red', 'green', 'blue'))(i))

    ax.set_xlim(xa.min(), xa.max())
    ax.set_ylim(xb.min(), xb.max())

    print("Saving " + image_path)

    os.makedirs(os.path.join(*[root_path, "images"]), exist_ok=True)
    fig.savefig(image_path, dpi=500)
    plt.close(fig)


def main():
    root_path = os.path.join(*[".", str(HIDDEN_LAYERS)])

    early_stopping = EarlyStopping(verbose=1,
                                   monitor="loss",
                                   patience=PATIENCE,
                                   min_delta=0.01,
                                   mode="min",
                                   restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(verbose=1,
                                       filepath=os.path.join(*[root_path, "weights", "{epoch:d}.weights.h5"]),
                                       save_weights_only=True,
                                       save_freq='epoch')
    model = get_model(HIDDEN_LAYERS)
    model.compile(optimizer=Adam(), loss=categorical_crossentropy)
    model.fit(x, to_categorical(y),
              verbose=2,
              batch_size=1,
              epochs=2048,
              callbacks=[early_stopping, model_checkpoint])

    pool = Pool(multiprocessing.cpu_count())
    pool.map_async(process_image, range(early_stopping.stopped_epoch - PATIENCE + 1)).get()
    pool.close()

    result_path = os.path.join(*[root_path, "result.mp4"])

    print("Processing " + result_path)

    frames = []
    size = (0, 0)

    for epoch in range(early_stopping.stopped_epoch - PATIENCE + 1):
        image = cv2.imread(os.path.join(*[root_path, "images", str(epoch + 1) + ".png"]))
        height, width, _ = image.shape
        size = (width, height)
        frames.append(image)

    out = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'mp4v'), 32, size)

    for frame in frames:
        out.write(frame)
    out.release()

    print("Saving " + result_path)


if __name__ == "__main__":
    main()
