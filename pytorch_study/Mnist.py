import torch
import pickle
import gzip
import requests
import numpy as np

from pathlib import Path
from matplotlib import pyplot


print(torch.__version__)

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL="http://deeplearning.net/data/mnist"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(f"{URL}/{FILENAME}").content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")

pyplot.show()
print(x_train.shape)