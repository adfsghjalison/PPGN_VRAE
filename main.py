import argparse
import tensorflow as tf
from model import vrnn
from flags import FLAGS

def run():
  with tf.device('/gpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    model = vrnn(FLAGS, sess)
    if FLAGS.mode == 'train':
        model.train()
    if FLAGS.mode == 'val':
        model.val()
    if FLAGS.mode == 'test':
        model.stdin_test()

if __name__ == '__main__':
    run()

