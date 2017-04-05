from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datetime import datetime
import os.path
import time
import numpy as np
import _init_paths
from datasets.factory import get_imdb
from networks.factory import get_network
from fast_rcnn.train import get_training_roidb
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from fast_rcnn.train import filter_roidb, SolverWrapper

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """parameter server jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")
tf.app.flags.DEFINE_string('worker_hosts', '',
                           """Comma-separated list of hostname:port for the """
                           """worker jobs. e.g. """
                           """'machine1:2222,machine2:1111,machine2:2222'""")

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 40000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            'Whether to log device placement.')

# Task ID is used to select the chief and also to access the local_step for
# each replica to check staleness of the gradients in sync_replicas_optimizer.
tf.app.flags.DEFINE_integer(
    'task_index', 0, 'Task ID of the worker/replica running the training.')

# More details can be found in the sync_replicas_optimizer class:
# tensorflow/python/training/sync_replicas_optimizer.py
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1,
                            """Number of gradients to collect before """
                            """updating the parameters.""")
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60,
                            'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180,
                            'Save summaries interval seconds.')

# **IMPORTANT**
# Please note that this learning rate schedule is heavily dependent on the
# hardware architecture, batch size and any changes to the model architecture
# specification. Selecting a finely tuned learning rate schedule is an
# empirical process that requires some experimentation. Please see README.md
# more guidance and discussion.
#
# Learning rate decay factor selected from https://arxiv.org/abs/1604.00981
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0,
                          'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94,
                          'Learning rate decay factor.')

tf.app.flags.DEFINE_string('imdb_name', None, '')
tf.app.flags.DEFINE_string('pretrained_model', None, '')
tf.app.flags.DEFINE_string('cfg', None, '')
tf.app.flags.DEFINE_string('network', None, '')
tf.app.flags.DEFINE_integer('max_iters', 70000, '')

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9  # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9  # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0  # Epsilon term for RMSProp.


def main(unused_args):
    assert FLAGS.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

    # Extract all the hostnames for the ps and worker jobs to construct the
    # cluster spec.
    ps_hosts = FLAGS.ps_hosts.split(',')
    worker_hosts = FLAGS.worker_hosts.split(',')
    tf.logging.info('PS hosts are: %s' % ps_hosts)
    tf.logging.info('Worker hosts are: %s' % worker_hosts)

    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                         'worker': worker_hosts})
    server = tf.train.Server(
        {'ps': ps_hosts,
         'worker': worker_hosts},
        job_name=FLAGS.job_name,
        task_index=FLAGS.task_index)

    imdb = get_imdb(FLAGS.imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print('Output will be saved to `{:s}`'.format(output_dir))

    network = get_network(FLAGS.network)
    print('Use network `{:s}` in training'.format(FLAGS.network))

    if FLAGS.job_name == 'ps':
        # `ps` jobs wait for incoming connections from the workers.
        server.join()
    else:
        # `worker` jobs will actually do the work.
        # assert dataset.data_files()
        # Only the chief checks for or creates train_dir.
        if FLAGS.task_index == 0:
            if not tf.gfile.Exists(FLAGS.train_dir):
                tf.gfile.MakeDirs(FLAGS.train_dir)
        train(network, imdb, roidb, output_dir, server.target, cluster_spec,
              pretrained_model=FLAGS.pretrained_model,
              max_iters=FLAGS.max_iters)


def train(network, imdb, roidb, output_dir, target, cluster_spec, pretrained_model=None, max_iters=None):
    num_workers = len(cluster_spec.as_dict()['worker'])
    num_parameter_servers = len(cluster_spec.as_dict()['ps'])

    if FLAGS.num_replicas_to_aggregate == -1:
        num_replicas_to_aggregate = num_workers
    else:
        num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate

    assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                           'num_parameter_servers'
                                                           ' must be > 0.')
    is_chief = (FLAGS.task_index == 0)

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % FLAGS.task_index,
            cluster=cluster_spec)):
        global_step = tf.contrib.framework.get_or_create_global_step()

        roidb = filter_roidb(roidb)
        saver = tf.train.Saver(max_to_keep=100)
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement)

        init_op = tf.global_variables_initializer()
        #sv = tf.train.Supervisor(is_chief=is_chief,
        #                         logdir=FLAGS.train_dir,
        #                         init_op=init_op,
        #                         summary_op=None,
        #                         global_step=global_step,
        #                         saver=saver,
        #                         save_model_secs=FLAGS.save_interval_secs)
        ## Get a session.
        #with sv.prepare_or_wait_for_session(target, config=sess_config) as sess:
        with tf.train.MonitoredTrainingSession(master=target, is_chief=is_chief, checkpoint_dir=FLAGS.train_dir) as sess:
            sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
            print('Solving...')
            sw.train_model(sess, max_iters)
            loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box = sw.compute_loss()
        pass


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
