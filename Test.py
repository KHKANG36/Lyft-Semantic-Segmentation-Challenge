import tensorflow as tf
import helper
import sys, skvideo.io, json, base64
import cv2
import numpy as np

batch_size = 300
num_classes = 3
image_input, keep_prob, logits, graph1 = helper.load('/tmp/frozen_graph_R3.pb')

def encode(array):
    array2 = cv2.resize(array, (800,600))
    retval, buffer = cv2.imencode('.png', array2)
    return base64.b64encode(buffer).decode("utf-8")

file = sys.argv[-1]
video = skvideo.io.vread(file)
answer_key = {}
frame = 1
tester = tf.nn.softmax(logits)

with tf.Session(graph=graph1) as sess:   

    for batch_i in range(0, len(video), batch_size):
        result=[]
        for rgb_frame in video[batch_i:batch_i+batch_size]:
            image=cv2.resize(rgb_frame, (256,192))
            result.append(image)

        im_softmax = sess.run([tester],{keep_prob: 1.0, image_input: result})         
        im_softmax_all = np.array(im_softmax).reshape(len(result), 192, 256, 3)

        for x in range(0,len(result)):
            segmentation_r = np.array(im_softmax_all[x,:, :,0])
            segmentation_r = (segmentation_r > 0.85).astype('uint8')

            segmentation_c = np.array(im_softmax_all[x,:, :,1])
            segmentation_c = (segmentation_c > 0.1).astype('uint8')
            
            answer_key[frame] = [encode(segmentation_c), encode(segmentation_r)]
            frame += 1
        
    print (json.dumps(answer_key))