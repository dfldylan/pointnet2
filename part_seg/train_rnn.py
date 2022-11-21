import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

# import pdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import part_dataset_all_normal
import datasets
from RNN import *

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=1, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)  # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model + '.py')
LOG_DIR = FLAGS.log_dir
RNN_LOG_DIR = 'rnn_log'
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 50
SEQ =5

# Shapenet official train/test split
DATA_PATH = os.path.join('/root/datasets/normal')
TRAIN_DATASET = datasets.RNNDataset(root=DATA_PATH, seq=SEQ)


# TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test')

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE,  # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        DECAY_STEP,  # Decay step.
        DECAY_RATE,  # Decay rate.
        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
        BN_INIT_DECAY,
        batch * BATCH_SIZE,
        BN_DECAY_DECAY_STEP,
        BN_DECAY_DECAY_RATE,
        staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):  #
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print "--- Get model and loss"
            # Get model and loss 
            pred, end_points, l3_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            print "--- Get training operator"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        # test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        # param init/restore
        if tf.train.get_checkpoint_state(LOG_DIR):
            print("Reading model parameters from %s" % LOG_DIR)
            saver.restore(sess, tf.train.latest_checkpoint(LOG_DIR))

        else:
            print("Created model with fresh parameters.")
            tf.global_variables_initializer().run(session=sess)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points,
               'final_points': l3_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer, saver)
            # eval_one_epoch(sess, ops, test_writer)

            # if epoch % 10 == 0:


def get_batch(dataset, idxs, start_idx, end_idx):
    ret = []
    for data in dataset[idxs[start_idx]]:
        ps, normal, seg = data
        batch_data = np.expand_dims(np.concatenate([ps, normal], axis=1), axis=0)
        batch_label = np.expand_dims(seg, axis=0)
        ret.append((batch_data, batch_label))
    return ret


def train_one_epoch(sess, ops, train_writer, saver):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) / BATCH_SIZE

    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    rnn_unit = 512
    temp = set(tf.global_variables())
    rnn = FluidRNN(SEQ-1, rnn_unit, 1024, 1024, lr=0.001)
    part_para = set(tf.global_variables()) - temp
    saver = tf.train.Saver(part_para, max_to_keep=1)
    if tf.train.get_checkpoint_state(RNN_LOG_DIR):
        print("Reading model parameters from %s" % RNN_LOG_DIR)
        saver.restore(sess, tf.train.latest_checkpoint(RNN_LOG_DIR))

    else:
        print("Created model with fresh parameters.")
        tf.variables_initializer(part_para).run(session=sess)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        ret = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        final_points_list = []
        for data in ret:
            batch_data, batch_label = data
            feed_dict = {ops['pointclouds_pl']: batch_data,
                         ops['labels_pl']: batch_label,
                         ops['is_training_pl']: False }
            final_points = sess.run([ops['final_points']], feed_dict=feed_dict)
            final_points_list.append(final_points)

        final_points_list = np.reshape(final_points_list, [-1, 1024])

        train_part = final_points_list[np.newaxis,:-1]
        test_part = final_points_list[np.newaxis, -1]

        loss, pred = rnn(sess, train_part, test_part)

        decode_op = True
        if decode_op:
            sess.run([ops[]])


        log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
        log_string('mean loss: %f' % (loss))
        save_path = saver.save(sess, os.path.join(RNN_LOG_DIR, "model.ckpt"))








        # Augment batched point clouds by rotation and jittering
        # aug_data = batch_data
        # aug_data = provider.random_scale_point_cloud(batch_data)
        # batch_data[:, :, 0:3] = provider.jitter_point_cloud(batch_data[:, :, 0:3])
        # pdb.set_trace()
        # train_writer.add_summary(summary, step)
        # pred_val = np.argmax(pred_val, 2)
        # correct = np.sum(pred_val == batch_label)
        # total_correct += correct
        # total_seen += (BATCH_SIZE*NUM_POINT)
        # loss_sum = loss_val

        # log_string(' -- %03d / %03d --' % (batch_idx + 1, num_batches))
        # log_string('mean loss: %f' % (loss_sum))

        # if (batch_idx + 1) % 100 == 0:
        #     # log_string('accuracy: %f' % (total_correct / float(total_seen)))
        #     # total_correct = 0
        #     # total_seen = 0
        #     # loss_sum = 0
        #     # Save the variables to disk.
        #     save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
        #     log_string("Model saved in file: %s" % save_path)


if __name__ == "__main__":
    log_string('pid: %s' % (str(os.getpid())))
    train()
    LOG_FOUT.close()
