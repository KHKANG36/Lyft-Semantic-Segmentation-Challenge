# import required functions
import os.path
import tensorflow as tf
import helper
import warnings
import cv2
import numpy as np
import time
from distutils.version import LooseVersion
import project_tests as tests
import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

# Define Variables
Reg = 0.001

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    
# Load VGG 
def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    # Load the model & weight 
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3, layer4, layer7

# Define the layers
def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    conv_1x1_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='SAME', 
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(Reg))
    deconv_layer7 = tf.layers.conv2d_transpose(conv_1x1_layer7, num_classes, 4, 2, padding='SAME', 
                                               kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(Reg))
    conv_1x1_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='SAME',
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(Reg))
    skip_connect1 = tf.add (deconv_layer7, conv_1x1_layer4)
    deconv_skip_connect1 = tf.layers.conv2d_transpose(skip_connect1, num_classes, 4, 2, padding='SAME', 
                                               kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(Reg))
    conv_1x1_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='SAME', 
                                       kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                       kernel_regularizer=tf.contrib.layers.l2_regularizer(Reg))
    skip_connect2 = tf.add (deconv_skip_connect1, conv_1x1_layer3)
    output = tf.layers.conv2d_transpose(skip_connect2, num_classes, 16, 8, padding='SAME', 
                                        kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                                               kernel_regularizer=tf.contrib.layers.l2_regularizer(Reg)) 
    
    return output

# Optimizer
def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1,num_classes), name="logits")
    labels = tf.reshape(correct_label, (-1,num_classes))
    cross_entropy_loss = tf.reduce_mean (tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    
    return logits, train_op, cross_entropy_loss

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    #Train function 
    for epoch in range(epochs):
        print("EPOCH: {}...".format(epoch+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0001})
        print("LOSS: {}".format(loss))
        
def run():
    num_classes = 3
    image_shape = (320, 480) #1/4 Resize
    data_dir = './data'
    runs_dir = './runs'
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth=True
    
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(data_dir, image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs = 20
        batch_size = 16
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output = layers(layer3, layer4, layer7, num_classes)
        #TF Placeholder for correct_label,learning_rate
        correct_label = tf.placeholder(dtype = tf.int32, shape = [None, None, None, num_classes])
        learning_rate = tf.placeholder(dtype = tf.float32)
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)
        
        # TODO: Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input, correct_label, keep_prob, learning_rate)
        
        #save model 
        saver = tf.train.Saver()
        saver.save(sess, "./lyft_challenge_R3.ckpt")
        # save graph
        tf.train.write_graph(tf.get_default_graph().as_graph_def(), '', './base_graph_R3.pb', as_text=False)
        # save "frozen" graph (variables are converted to constants)
        output_node_names = 'logits'
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(),output_node_names.split(",")) 
        tf.train.write_graph(output_graph_def, '', './frozen_graph_R3.pb', as_text=False)
        print("# of operations in the final graph=", len(output_graph_def.node))   

run()