from __future__ import print_function
import tensorflow as tf
import os

# Dataset Parameters - CHANGE HERE
DATASET_PATH = '../data/TRANCOS_v3/'

# Image Parameters
IMG_HEIGHT = 112
IMG_WIDTH = 200
CHANNELS = 3

# Reading the dataset
def read_images(dataset_path, batch_size, mode):
    image_paths, labels = [], []
        # Read dataset file
    if mode == 'train':
        dataset_path_mod = dataset_path + 'image_sets/trainval.txt'
    elif mode == 'test':
        dataset_path_mod = dataset_path + 'image_sets/test.txt'
        
    with open(dataset_path_mod, 'r') as f:
        data = f.read().splitlines()
        
    for d in data:
        image_paths.append(dataset_path+'images/'+d)
        label_path = dataset_path+'images/'+d.replace('.jpg','.txt')
        
        with open(label_path, 'r', encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
            
        labels.append(i+1)

    # Convert to Tensor
    imagepaths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    labels     = tf.convert_to_tensor(labels, dtype=tf.float32)
    # Build a TF Queue, shuffle data
    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    # Read images from disk
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    # Resize images to a common size
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])

    # Normalize
    image = tf.image.per_image_standardization(image) #better ?
    # Create batches
    X, Y = tf.train.batch([image, label], batch_size=batch_size)
                          #num_threads=4) ############################################################# num threads?

    return X, Y

# Parameters
learning_rate = 0.001
num_steps = 500
batch_size = 8
display_step = 100
dropout = 0.2

# Build the data input
X_train_batch, Y_train_batch = read_images(DATASET_PATH, batch_size, mode='train')
X_test_batch, Y_test_batch   = read_images(DATASET_PATH, batch_size, mode='test')

# Create model
def conv_net(x,reuse=False, dropout=0.1, kernel_size=5, filters=32, hiddenunits=100):
    # Define a scope for reusing the variables
    with tf.variable_scope('model', reuse=reuse):
        conv1    = tf.layers.conv2d(inputs=x,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    activation=tf.nn.relu)
        maxpool1 = tf.layers.max_pooling2d(inputs=conv1,
                                           pool_size=2,
                                           strides=2)
        conv2    = tf.layers.conv2d(inputs=maxpool1,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    activation=tf.nn.relu)
        maxpool2 = tf.layers.max_pooling2d(inputs=conv2,
                                           pool_size=2,
                                           strides=2)
        conv3    = tf.layers.conv2d(inputs=maxpool2,
                                    filters=filters,
                                    kernel_size=kernel_size,
                                    activation=tf.nn.relu)
        maxpool3 = tf.layers.max_pooling2d(inputs=conv3,
                                           pool_size=2,
                                           strides=2)
        flat     = tf.contrib.layers.flatten(inputs=maxpool3)
        dropout1 = tf.layers.dropout(inputs=flat,
                                     rate=dropout,
                                     training=True)
        dense1   = tf.layers.dense(inputs=dropout1,
                                   units=hiddenunits,
                                   activation=tf.nn.relu)
        
        dropout2 = tf.layers.dropout(inputs=dense1,
                                     rate=dropout,
                                     training=True)
        out = tf.layers.dense(dropout2,
                              units=1)
    #simple linear regression model
    #hidden = tf.layers.dense(inputs=x,
    #                         units=100,
    #                         activation=tf.nn.relu)
    #out = tf.layers.dense(hidden, 1)

    return out

# Create a graph for training
logits_train = conv_net(X_train_batch, reuse=False)
# Create another graph for testing that reuse the same weights
logits_test = conv_net(X_test_batch, reuse=True)

# Define loss and optimizer
train_loss = tf.reduce_mean(tf.square(tf.subtract(Y_train_batch, logits_train))) #mean square error
test_loss  = tf.reduce_mean(tf.square(tf.subtract(Y_test_batch, logits_test)))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op  = optimizer.minimize(train_loss)

# Saver object
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(tf.global_variables_initializer())
    # initialize the queue threads
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # Training
    for step in range(num_steps):
        if step % display_step == 0:
            # Run optimization and calculate batch test and train loss
            _, batchtrainloss, batchtestloss = sess.run([train_op, train_loss, test_loss])
            print('Step {}, Train batch loss: {:.3f}, Test batch loss: {:.3f}'.format(step, batchtrainloss, batchtestloss))
        else:
            # Only run backprop
            sess.run(train_op)
            
    coord.request_stop()
    coord.join(threads)
    
    # Save the model
    save_path = saver.save(sess, "/tmp/model.ckpt")
    #saver.save(sess, 'my_tf_model')
