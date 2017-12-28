from __future__ import print_function
import tensorflow as tf
import os
from hyperopt import fmin, tpe, hp
from data_loader_cnn import read_images, conv_net

# Dataset Parameters
DATASET_PATH = '../data/TRANCOS_v3/'

# Image Parameters
N_CLASSES = 1
IMG_HEIGHT = 112
IMG_WIDTH = 200
CHANNELS = 3


# Parameters
num_steps = 1
X_train_batch, Y_train_batch = read_images(DATASET_PATH, batch_size=10, mode='train')
X_test_batch, Y_test_batch   = read_images(DATASET_PATH, batch_size=10, mode='test')
X_val_batch, Y_val_batch     = read_images(DATASET_PATH, batch_size=10, mode='validation')

def f(space):
    # Build the data input
    # Create a graph for training
    logits_train = conv_net(X_train_batch,
                            dropout=space['dropout'],
                            kernel_size=space['kernel_size'],
                            filters=32,
                            hiddenunits=500)
    # Create another graph for testing that reuse the same weights
    #logits_test = conv_net(X_test_batch)
    
    # Define loss and optimizer (with train logits, for dropout to take effect)
    loss_op   = tf.reduce_mean(tf.square(tf.subtract(Y_train_batch, logits_train))) #mean square error
    optimizer = tf.train.AdamOptimizer(learning_rate=space['learning_rate'])
    train_op  = optimizer.minimize(loss_op)
    trainloss_runner = 0
    # Evaluate model (with test logits, for dropout to be disabled)
    #test_loss = tf.reduce_mean(tf.square(tf.subtract(Y_batch_train, logits_test)))
    with tf.Session() as sess:
        # Run the initializer
        sess.run(tf.global_variables_initializer())
        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
    
        # Training cycle
        for step in range(1, num_steps+1):
    
            if step % 100 == 0:
                # Run optimization and calculate batch loss and accuracy
                _, trainloss = sess.run([train_op, loss_op])
                trainloss_runner = trainloss
                print("Step " + str(step) + ", Train minibatch Loss= " + \
                      "{:.4f}".format(trainloss))
                
                #_, testloss = sess.run([test_op, ])
            else:
                # Only run the optimization op (backprop)
                sess.run(train_op)
            
    coord.request_stop()
    coord.join(threads)
  
    return(trainloss_runner) #thing to be minimized


    
space = {'learning_rate': hp.uniform('lr', 0.0001, 0.01), ############################## crucial
         'dropout': hp.uniform('dr', 0.1, 0.5),
         'kernel_size' : hp.choice('kernel', [3,5,7,9])}
         
#         'batch_size': hp.choice('bs', [32, 64, 128]),

best = fmin(
    fn=f,
    space=space,
    algo=tpe.suggest,
    max_evals = 2)  ################################################################## crucial)

print("Found minimum with parameters:")
print(best)

with open('hyperopt_best.txt', 'w') as f:
    f.write(str(best))