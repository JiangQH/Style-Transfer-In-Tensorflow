import tensorflow as tf
import os.path as osp
import os
import losses
import model
from dataset import Dataset
from vgg import Vgg
from util import preprocess

def main(Config):
    # get the style feature
    style_features = losses.get_style_feature(Config)
    # prepare some dirs for use
    model_dir = 'model'
    if not osp.exists(model_dir):
        os.mkdir(model_dir)

    # construct the graph and model
    with tf.Graph().as_default():
        # prepare the dataset
        images = Dataset(Config).imagedata_pipelines()
        # the trainnet
        generated = model.inference_trainnet(images)
        # concat the content image and the generated together to save time and feed to the vgg net one time
        # preprocess the generated
        preprocess_generated = preprocess(generated, Config)
        layer_infos = Vgg(Config.model_path).build(tf.concat([preprocess_generated, images], 0))
        # get the loss
        content_loss = losses.content_loss(layer_infos, Config.content_layers)
        style_loss = losses.style_loss(layer_infos, Config.style_layers, style_features)
        tv_loss = losses.tv_loss(generated)
        loss = Config.style_weight * style_loss + Config.content_weight * content_loss + Config.tv_weight * tv_loss
        tf.add_to_collection('losses', loss)
        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        # variable init
        variable_init = tf.global_variables_initializer()
        
