import tensorflow as tf
import os.path as osp
import os
import losses
import model
from dataset import Dataset
from vgg import Vgg
from util import preprocess, load_config
import numpy as np
from PIL import Image
import argparse


def solve(Config):
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

        # add summary
        tf.summary.scalar('content_loss', content_loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('tv_loss', tv_loss)
        tf.summary.scalar('weighted_content_loss', content_loss * Config.content_weight)
        tf.summary.scalar('weighted_style_loss', style_loss * Config.style_weight)
        tf.summary.scalar('weighted_tv_loss', tv_loss * Config.tv_weight)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.image('generated', generated)
        summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(model_dir)


        # variable init
        init_op = tf.global_variables_initializer()

        # train op
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(Config.lr).minimize(total_loss, global_step=global_step)
        # no restore during train

        # begin the training work
        sess = tf.Session()
        sess.run(init_op)
        saver = tf.train.Saver(tf.trainable_variables())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in xrange(Config.max_iter):
            _, loss_value, style_loss_value, content_loss_value, gen = sess.run([train_op, loss,
                                                                            Config.style_weight * style_loss,
                                                                            Config.content_weight * content_loss,
                                                                            generated])
            if step % Config.display == 0:
                print "{}[iterations], train loss {}".format(step, loss_value)
                assert not np.isnan(loss_value), 'model with loss nan'
            if step % Config.snapshot == 0:
                # save the generated to see
                print 'adding summary and saving snapshot...'
                im = Image.fromarray(np.uint8(gen))
                save_name = osp.join(model_dir, str(step)+'.png')
                im.save(save_name)
                saver.save(sess, osp.join(model_dir, 'model.ckpt'), global_step=step)
                summary_str = sess.run(summary)
                writer.add_summary(summary_str, global_step=step)
                writer.flush()

        coord.request_stop()
        coord.join(threads)
        sess.close()

        print 'done'


def main(argv=None):
    print 'begin training'
    paser = argparse.ArgumentParser()
    paser.add_argument('-c', '--conf', help='path to the config file')
    args = paser.parse_args()
    Config = load_config(args)
    solve(Config)


if __name__ == '__main__':
    tf.app.run()