# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatasetReader as dataset
import cv2
import os
from six.moves import xrange

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', '2', 'batch size for training/testing')
tf.flags.DEFINE_string('logs_dir', 'logs/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'Data_zoo/MIT_SceneParsing/', 'path to dataset')
tf.flags.DEFINE_string('data_name', 'ADEChallengeData2016', 'data name')
tf.flags.DEFINE_float('learning_rate', '1e-5', 'initial learning rate for Adam Optimizer')
tf.flags.DEFINE_string('model_path', 'Model_zoo/imagenet-vgg-verydeep-19.mat', 'path to vgg FLAGS.model mat')
tf.flags.DEFINE_bool('debug', 'False', 'debug mode: True/False')
tf.flags.DEFINE_string('mode', 'train', 'mode train/visualize')
tf.flags.DEFINE_float('weight_decay', '1e-3', 'L2 regularization, decay=0.0 means no L2')

tf.flags.DEFINE_integer('MAX_ITERATION', '100001', 'upper limit of iterations')
tf.flags.DEFINE_integer('NUM_OF_CLASSES', '3', 'number of classes detected including background = # of real class + 1')
tf.flags.DEFINE_integer('IMAGE_SIZE', '224',
                        'image size for height and weight at the same time. Suggested size is 32 times, 224/32 = 7')


# batch_size = 1  # batch 大小
# logs_dir = "logs/"
# data_dir = "Data_zoo/MIT_SceneParsing/"  # 存放数据集的路径，需要提前下载
# data_name = "ADEChallengeData2016"
# learning_rate = 1e-5  # 学习率
# model_path = "Model_zoo/imagenet-vgg-verydeep-19.mat"  # VGG网络参数文件，需要提前下载
# debug = False
# mode = 'visualize'  # 训练模式train | visualize
#
# modeL_URL = 'http://www.vlfeat.org/matconvnet/FLAGS.models/beta16/imagenet-vgg-verydeep-19.mat'  # 训练好的VGGNet参数
#
# MAX_ITERATION = int(1e5 + 1)  # 最大迭代次数
# NUM_OF_CLASSES = 3  # 类的个数
# IMAGE_SIZE = 224  # 图像尺寸


# 根据载入的权重建立原始的 VGGNet 的网络
def vgg_net(weights, image):
    """
        weights ????
        image ????
    """

    # 2+2+4+4+4 = 16 : VGG-19:16 conv + 3 fully connected layer
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image

    for i, name in enumerate(layers):
        kind = name[:4]  # conv,relu,pool

        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]

            kernels = utils.get_variable(np.transpose(kernels, [1, 0, 2, 3]), name=name + '_w')
            bias = utils.get_variable(bias.reshape(-1), name=name + '_b')
            current = utils.conv2d_basic(current, kernels, bias, FLAGS.weight_decay)

            print('current shape: ', np.shape(current))
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
            print('current shape: ', np.shape(current))
        net[name] = current

    return net


# FCN的网络结构定义，网络中用到的参数是迁移VGG训练好的参数
def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    : param image: input image, should have values in range 0-255
    : param keep_prob
    : return:
    """

    # loading FLAGS.model 
    print('original image', np.shape(image))

    model_data = utils.get_model_data(FLAGS.model_path)

    # mean = model_data['normalization'][0][0][0]
    # mean_pixel = np.mean(mean, axis=(0, 1))  # array([123.68 , 116.779, 103.939])

    mean_pixel = np.array([134.69] * 3)

    weights = np.squeeze(model_data['layers'])

    # image preprocessing
    processed_image = utils.process_image(image, mean_pixel)
    print('preprocessed image: ', np.shape(processed_image))

    with tf.variable_scope('inference'):
        # build original VGGNET-19
        print('start building VGG')
        image_net = vgg_net(weights, processed_image)

        # 在VGGNet-19之后添加 一个池化层和三个卷积层    
        conv_final_layer = image_net['conv5_3']
        print("VGG处理后的图像：", np.shape(conv_final_layer))

        pool5 = utils.max_pool_2x2(conv_final_layer)  # batch x 7 x7 x 512

        print('pool5', np.shape(pool5))

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name='b6')
        conv6 = utils.conv2d_basic(pool5, W6, b6, FLAGS.weight_decay)
        relu6 = tf.nn.relu(conv6, name='relu6')

        if FLAGS.debug:
            utils.add_activation_summary(relu6)

        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        print('conv6:', np.shape(relu_dropout6))

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7, FLAGS.weight_decay)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        print("conv7:", np.shape(relu_dropout7))

        W8 = utils.weight_variable([1, 1, 4096, FLAGS.NUM_OF_CLASSES], name="W8")
        b8 = utils.bias_variable([FLAGS.NUM_OF_CLASSES], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8, FLAGS.weight_decay)

        print("conv8:", np.shape(conv8))  # batch x 7 x 7 x 151
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # 对卷积后的结果进行反卷积操作

        deconv_shape1 = image_net["pool4"].get_shape()  # 14x14x512
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, FLAGS.NUM_OF_CLASSES], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1,
                                                 output_shape=tf.shape(image_net["pool4"]))  # 14x14x512
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        print("pool4 and de_conv8 ==> fuse1:", np.shape(fuse_1))  # (14, 14, 512)

        deconv_shape2 = image_net["pool3"].get_shape()  # 28x28x256
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value],
                                     name="W_t2")  # [4,4,256,512].
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")  # 256
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2,
                                                 output_shape=tf.shape(image_net["pool3"]))  # 28x28x256
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        print("pool3 and deconv_fuse1 ==> fuse2:", np.shape(fuse_2))  # (28, 28, 256)

        shape = tf.shape(image)  # 1x224x224x3
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], FLAGS.NUM_OF_CLASSES])  # 224x224X3X151
        W_t3 = utils.weight_variable([16, 16, FLAGS.NUM_OF_CLASSES, deconv_shape2[3].value],
                                     name="W_t3")  # [16,16,151,256]
        b_t3 = utils.bias_variable([FLAGS.NUM_OF_CLASSES], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        print("conv_t3:", [np.shape(image)[1], np.shape(image)[2], FLAGS.NUM_OF_CLASSES])  # (224,224,151)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")  # (224,224,1)

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


# 返回优化器
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


# 主函数,返回优化器的操作步骤http://www.vlfeat.org/matconvnet/FLAGS.models/beta16/imagenet-vgg-verydeep-19.mat
def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, FLAGS.IMAGE_SIZE, FLAGS.IMAGE_SIZE, 1], name="annotation")

    print("setting up vgg initialized conv layers ...")

    # 定义好FCN的网络模型
    pred_annotation, logits = inference(image, keep_probability)

    labels = tf.squeeze(annotation,
                        squeeze_dims=[3])

    # 定义损失函数，这里使用交叉熵的平均值作为损失函数
    model_loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                labels=labels,
                                                                                name="entropy")))
    model_loss_summary = tf.summary.scalar("model entropy", model_loss)

    tf.add_to_collection('losses', model_loss)
    loss = tf.add_n(tf.get_collection('losses'))  # including L2 losses

    loss_summary = tf.summary.scalar("total entropy", loss)
    # 定义优化器 
    trainable_var = tf.trainable_variables()
    # for _ in trainable_var:
    #     print(_)

    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    # 加载数据集
    print("Setting up image reader...")

    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir, FLAGS.data_name)

    print("训练集的大小:", len(train_records))
    print("验证集的大小:", len(valid_records))

    print("Setting up dataset reader")

    image_options = {'resize': True, 'resize_size': FLAGS.IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    # 开始训练模型
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    print("Setting up Saver...")
    saver = tf.train.Saver()

    # summary merging
    summary_merge = tf.summary.merge_all()

    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('model ' + ckpt.model_checkpoint_path.split('-')[-1] + ' restored...')

    start = 0
    if ckpt:
        start = ckpt.model_checkpoint_path.split('-')[-1]
        start = int(start) + 1

    # best valid
    best = 1e8  # set as infi

    if FLAGS.mode == "train":
        for itr in range(start, FLAGS.MAX_ITERATION):
            train_images, train_annotations, _ = train_dataset_reader.next_batch(FLAGS.batch_size)
            print(np.shape(train_images), np.shape(train_annotations))
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # lo, ls, anno, lab = sess.run([loss, logits, pred_annotation, labels], feed_dict=feed_dict)
            # nan = np.sum(np.isnan(ls))
            # print('nan #', nan)
            # print('labels:', lab, 'unique', np.unique(lab))
            # print('ls', ls)
            # print('shape', ls.shape)
            # print('pred_annotation', anno)
            # print('loss', lo)
            # print('*' * 30)
            sess.run(train_op, feed_dict=feed_dict)
            print("step:", itr)

            if itr % 10 == 0:
                if FLAGS.debug:
                    train_loss, summary_str = sess.run([loss, summary_merge], feed_dict=feed_dict)
                else:
                    train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 100 == 0:
                valid_images, valid_annotations, _ = validation_dataset_reader.next_batch(14)  # FLAGS.batch_size)
                valid_loss, summary_sva = sess.run([model_loss, model_loss_summary],
                                                   feed_dict={image: valid_images, annotation: valid_annotations,
                                                              keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)

                # saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

                if valid_loss < best:
                    best = valid_loss
                    # os.system('rm -f ./logs/best/*')
                    saver.save(sess, FLAGS.logs_dir + "best/" + "model.ckpt", itr)
    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations, idx = validation_dataset_reader.next_batch(
            FLAGS.batch_size)  # get_random_batch(FLAGS.batch_size)

        # testing for any size
        # valid_images = np.expand_dims(valid_images[0], axis=0)
        # valid_annotations = np.expand_dims(valid_annotations[0], axis=0)
        # print(valid_images.shape)
        # print(valid_annotations.shape)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)
        for itr in range(FLAGS.batch_size):
            # utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5 + itr))
            # utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5 + itr))
            # utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5 + itr))
            valid_image = valid_images[itr].astype(np.uint8)
            utils.save_image(valid_image, FLAGS.logs_dir, name="inp_" + idx[itr])

            predone = pred[itr].astype(np.uint8)
            img_rgb = utils.color_transform(predone, FLAGS.NUM_OF_CLASSES)
            utils.save_image(img_rgb.astype(np.uint8), FLAGS.logs_dir, name="pred_color_" + idx[itr])

            dst = utils.sobel(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
            edgedImage = utils.draw_edge(dst, valid_image)
            utils.save_image(edgedImage.astype(np.uint8), FLAGS.logs_dir, name="pred_" + idx[itr])
            print("Saved image: %d" % itr)

    train_writer.close()
    validation_writer.close()
    sess.close()


if __name__ == "__main__":
    tf.app.run()
