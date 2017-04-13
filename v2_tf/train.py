import tensorflow as tf
import os.path as osp
import os
import losses
import model
from dataset import Dataset
from vgg import Vgg
from util import preprocess
import time


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
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        try:
            while not coord.should_stop():
                _, loss, step = sess.run([train_op, total_loss, global_step])
                if step % Config.disply == 0:
                    eps = time.time() - start_time
                    start_time = time.time()
                    print 'time consumes {}, step: {}, total_loss {}'.format(eps, step, loss)
                if step % Config.summary == 0:
                    print 'adding summary...'
                    summary_str = sess.run(summary)
                    writer.add_summary(summary_str)
                    writer.flush()
                if step % Config.snapshot == 0:
                    saver.save(sess, osp.join(model_dir, 'model.ckpt'), global_step=step)
        except tf.errors.OutOfRangeError:
            saver.save(sess, osp.join(model_dir, 'model.ckpt'))
            print 'epoch limit arrived'
        finally:
            coord.request_stop()
        coord.join(threads)

