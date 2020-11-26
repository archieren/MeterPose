# 兼容tf-1.14
import tensorflow as tf
import numpy as np

INPUT_NAME = 'input_images'
OUTPUT_NODE_NAMES = OUTPUT_NAME = 'resnet_model/heatmaps'

def load_a_frozen_graph(path_to_frozen_graph):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.io.gfile.GFile(path_to_frozen_graph, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    graph = tf.Graph()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(graph=graph, config=config)

    with graph.as_default():
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name='')

    return graph, sess

class python_model():
    def __init__(self, model_path, i_name=INPUT_NAME, o_name=OUTPUT_NAME):
        self.__graph, self.__sess = load_a_frozen_graph(model_path)
        self.__input = self.__graph.get_tensor_by_name(i_name + ':0')
        self.__fetches = {o_name: self.__graph.get_tensor_by_name(o_name + ':0')}

    def inference(self, image):
        image = image.resize((224, 224))
        feeds = {self.__input: np.expand_dims(image, 0)}
        output = self.__sess.run(fetches=self.__fetches, feed_dict=feeds)
        return output[OUTPUT_NAME][0]
