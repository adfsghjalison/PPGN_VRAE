import tensorflow as tf

tf.app.flags.DEFINE_string('mode','train', 'train / test / stdin')
tf.app.flags.DEFINE_string('model_dir','model', 'output model dir')
tf.app.flags.DEFINE_string('data_dir','data', 'data dir')
tf.app.flags.DEFINE_string('data_name','NLPCC', 'data dir')
tf.app.flags.DEFINE_string('feed_previous', True, 'whether feed previous')
tf.app.flags.DEFINE_string('KL_annealing', True, 'whether do KL annealing')
tf.app.flags.DEFINE_integer('batch_size', 200, 'batch size')
tf.app.flags.DEFINE_integer('latent_dim', 500, 'laten size')
tf.app.flags.DEFINE_integer('sequence_length', 15, 'sentence length')

tf.app.flags.DEFINE_integer('printing_step', 1, 'saving step')
tf.app.flags.DEFINE_integer('saving_step', 2, 'saving step')
tf.app.flags.DEFINE_integer('num_steps', 6, 'number of steps')

"""
tf.app.flags.DEFINE_integer('printing_step', 1000, 'saving step')
tf.app.flags.DEFINE_integer('saving_step', 20000, 'saving step')
tf.app.flags.DEFINE_integer('num_steps', 100000, 'number of steps')
"""

FLAGS = tf.app.flags.FLAGS

FLAGS.data_dir = os.path.join(FLAGS.data_dir, 'data_{}'.format(FLAGS.data_name))
FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'model_{}'.format(FLAGS.data_name))

if not os.path.exists(FLAGS.model_dir):
  os.mkdir(FLAGS.model_dir)
  print ('Create model dir : {}'.format(FLAGS.model_dir))

BOS = 0
EOS = 1
UNK = 2
DROPOUT = 3

