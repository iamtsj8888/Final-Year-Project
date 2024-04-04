import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
import scipy.io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from model import build_model

global image_h
global image_w

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.1):
    """ Loading the images and masks """
    X = sorted(glob(os.path.join(path, "images", "*.jpg")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    """ Spliting the data into training and testing """
    split_size = int(len(X) * split)

    train_x, valid_x = train_test_split(X, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(Y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y)

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("Model/files")

    """ Hyperparameters """
    image_h = 512
    image_w = 512
    input_shape = (image_h, image_w, 3)
    batch_size = 4
    lr = 1e-4
    num_epochs = 100

    """ Paths """
    dataset_path = "Model/people_segmentation"
    model_path = os.path.join("Model/files", "model.h5")
    csv_path = os.path.join("Model/files", "data.csv")

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y) = load_dataset(dataset_path, split=0.2)
    print(f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)}")
    print("")