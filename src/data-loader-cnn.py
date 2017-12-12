from __future__ import print_function
import tensorflow as tf
import os

# Dataset Parameters - CHANGE HERE
DATASET_PATH = '../data/TRANCOS_v3/'

# Image Parameters
N_CLASSES = 1
IMG_HEIGHT = 112
IMG_WIDTH = 200
CHANNELS = 3


# Reading the dataset
def read_images(dataset_path, batch_size, mode):
    imagepaths, labels = [], []
        # Read dataset file
    if mode == 'train':
        dataset_path = dataset_path + 'image_sets/training.txt'
    elif mode == 'validation':
        dataset_path = dataset_path + 'image_sets/validation.txt'
    elif mode == 'test':
        dataset_path = dataset_path + 'image_sets/test.txt'
        
    with open(dataset_path, 'r') as f:
        data = f.read().splitlines()
        
    for d in data:
        imagepaths.append(d.split(' ')[0])
        
        
        labels.append(1)

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = image * 1.0/127.5 - 1.0

    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, Y

# Parameters
learning_rate = 0.001
num_steps = 1000
batch_size = 32
display_step = 100


# Build the data input
X_train, Y_train = read_images(DATASET_PATH, batch_size, mode='train')
