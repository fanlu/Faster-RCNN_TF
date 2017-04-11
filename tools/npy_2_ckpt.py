import tensorflow as tf
import numpy as np
import argparse
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', dest='data_path', help='',
                        default='npy', type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    data_dict = np.load(args.data_path).item()
    for key in data_dict:
        with tf.variable_scope(key, reuse=True):
            for subkey in data_dict[key]:
                try:
                    var = tf.get_variable(subkey)
                    var.assign(data_dict[key][subkey])
                    tf.add_to_collection(subkey, var)
                except ValueError:
                    print "ignore"

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, 'my-model')
