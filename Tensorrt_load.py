from tensorflow.python.platform import gfile
import tensorflow as tf
from tensorflow.contrib import tensorrt as trt

graph_filename ='frozen_graph_R3.pb'
f = gfile.FastGFile(graph_filename, 'rb')

# define graph def object
frozen_graph_def = tf.GraphDef()

# store frozen graph from pb file
frozen_graph_def.ParseFromString(f.read())

# Parameters:
output_node_name = "logits"
workspace_size = 1 << 30
precision = "FP32"
batch_size = 1

trt_graph = trt.create_inference_graph(
                frozen_graph_def,
                [output_node_name],
                max_batch_size=batch_size,
                max_workspace_size_bytes=workspace_size,
                precision_mode=precision)


# write modified graph def to disk
graph_filename_converted = 'frozen_graph_R3_tensorrt.pb'


with gfile.FastGFile(graph_filename_converted, 'wb') as s:
    s.write(trt_graph.SerializeToString())