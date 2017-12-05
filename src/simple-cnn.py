import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt

def readData(entries):
    training_imgs = [line.rstrip('\n') for line in open('../data/TRANCOS_v3/image_sets/training.txt')]
    img_path = '../data/TRANCOS_v3/images/'
    imgs = np.zeros([entries, 480, 640, 3]) # images in np array format
    labels = np.zeros([entries]) # number of cars in each image
    for i in range(entries):
        imgs[i] = plt.imread(str(img_path) + str(training_imgs[i]))
        with open(str(img_path) + str(training_imgs[i].split(".")[0]) + ".txt") as f:
            for c, l in enumerate(f):
                pass
        labels[i] = c+1

    return imgs, labels

def splitData(data):
    pass

class model:

    X = tf.placeholder(tf.float32, [None, 480, 640, 3])
    Y = tf.placeholder(tf.float32, [None])

    # weights init
    W_conv_1 = tf.Variable(tf.truncated_normal([5,5,3,4], stddev=0.1))
    b_conv_1 = tf.Variable(tf.ones([4]))

    W_dense_1 = tf.Variable(tf.truncated_normal([3,3], stddev=0.1))
    b_dense_1 = tf.Variable(tf.ones([100]))

    # model architecture
    #conv1 = tf.nn.relu(tf.nn.conv2d(X, W_conv_1, strides=[1,1,1,1], padding="SAME") + b_conv_1) # (?,  28, 28, 4)
def main():

    # 1. read data
    X_train, Y_train = readData(100)
    print(X_train.shape)

    # 3 init model
    our_model = model()


if __name__ == "__main__":
    main()