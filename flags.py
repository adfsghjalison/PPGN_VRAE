import tensorflow as tf

tf.app.flags.DEFINE_string('mode','train', 'train / test / stdin')
tf.app.flags.DEFINE_string('model_dir','model', 'output model dir')
tf.app.flags.DEFINE_string('data_dir','data', 'data dir')
tf.app.flags.DEFINE_string('feed_previous', True, 'whether feed previous')
tf.app.flags.DEFINE_string('KL_annealing', True, 'whether do KL annealing')
tf.app.flags.DEFINE_integer('batch_size', 200, 'batch size')
tf.app.flags.DEFINE_integer('latent_dim', 500, 'laten size')
tf.app.flags.DEFINE_integer('printing_step', 1000, 'saving step')
tf.app.flags.DEFINE_integer('saving_step', 20000, 'saving step')
tf.app.flags.DEFINE_integer('num_steps', 100000, 'number of steps')
tf.app.flags.DEFINE_integer('sequence_length', 10, 'sentence length')

FLAGS = tf.app.flags.FLAGS
BOS = 0
EOS = 1
UNK = 2
DROPOUT = 3

