from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import sys
import pprint
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


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')

    parser.add_argument('--job_name', dest='job_name', help='One of "ps", "worker"',
                        default='ps', type=str)
    parser.add_argument('--task_index', dest='task_index', help='Task ID of the worker/replica running the training.',
                        default=0, type=int)
    parser.add_argument('--ps_hosts', dest='ps_hosts', help="""Comma-separated list of hostname:port for the """
                                                            """parameter server jobs. e.g. """
                                                            """'machine1:2222,machine2:1111,machine2:2222'""",
                        default='localhost:1120', type=str)
    parser.add_argument('--worker_hosts', dest='worker_hosts', help="""Comma-separated list of hostname:port for the """
                                                                    """worker jobs. e.g. """
                                                                    """'machine1:2222,machine2:1111,machine2:2222'""",
                        default='localhost:1121', type=str)

    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


def train(network, imdb, roidb, output_dir, target, cluster_spec, pretrained_model=None, max_iters=None):
    num_workers = len(cluster_spec.as_dict()['worker'])
    num_parameter_servers = len(cluster_spec.as_dict()['ps'])

    # if args.num_replicas_to_aggregate == -1:
    #    num_replicas_to_aggregate = num_workers
    # else:
    #    num_replicas_to_aggregate = args.num_replicas_to_aggregate

    assert num_workers > 0 and num_parameter_servers > 0, (' num_workers and '
                                                           'num_parameter_servers'
                                                           ' must be > 0.')
    is_chief = (args.task_index == 0)

    with tf.device(tf.train.replica_device_setter(
            worker_device="/job:worker/task:%d" % args.task_index,
            cluster=cluster_spec)):
        global_step = tf.contrib.framework.get_or_create_global_step()

        roidb = filter_roidb(roidb)
        saver = tf.train.Saver(max_to_keep=100)
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)

        init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(is_chief=is_chief,
                                 logdir=output_dir,
                                 init_op=init_op,
                                 summary_op=None,
                                 global_step=global_step,
                                 saver=saver)
        ## Get a session.
        sess = sv.prepare_or_wait_for_session(target, config=sess_config)
        # with tf.train.MonitoredTrainingSession(master=target, is_chief=is_chief,
        #                                        checkpoint_dir=output_dir) as sess:
        # with tf.Session(target=target, config=sess_config) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print('Solving...')
        sw.train_model(sess, max_iters)
        # loss, cross_entropy, loss_box, rpn_cross_entropy, rpn_loss_box = sw.compute_loss()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    assert args.job_name in ['ps', 'worker'], 'job_name must be ps or worker'

    # Extract all the hostnames for the ps and worker jobs to construct the
    # cluster spec.
    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')
    tf.logging.info('PS hosts are: %s' % ps_hosts)
    tf.logging.info('Worker hosts are: %s' % worker_hosts)

    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts,
                                         'worker': worker_hosts})
    server = tf.train.Server(
        {'ps': ps_hosts,
         'worker': worker_hosts},
        job_name=args.job_name,
        task_index=args.task_index)

    imdb = get_imdb(args.imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    print('Output will be saved to `{:s}`'.format(output_dir))

    network = get_network(args.network_name)
    print('Use network `{:s}` in training'.format(args.network_name))

    if args.job_name == 'ps':
        # `ps` jobs wait for incoming connections from the workers.
        server.join()
    else:
        # `worker` jobs will actually do the work.
        # assert dataset.data_files()
        # Only the chief checks for or creates train_dir.
        if args.task_index == 0:
            if not tf.gfile.Exists(output_dir):
                tf.gfile.MakeDirs(output_dir)
        train(network, imdb, roidb, output_dir, server.target, cluster_spec,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
