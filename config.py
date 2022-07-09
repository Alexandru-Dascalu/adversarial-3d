import tensorflow as tf

flags = tf.compat.v1.flags

############################
#    hyper parameters      #
############################

flags.DEFINE_integer('batch_size', 12, 'batch size')
flags.DEFINE_integer('iterations', 1000, 'iterations')
flags.DEFINE_float('learning_rate', 0.002, 'learning rate')
flags.DEFINE_float('l2_weight', 0.001, 'the weighting factor for l2 loss')
# 463 - broom
flags.DEFINE_integer('target', 463, 'the label for adversarial examples')

############################
#   environment setting    #
############################

flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('model_dir', 'model_dir', 'model directory')
flags.DEFINE_string('model_url', 'http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz', 'model download link')
flags.DEFINE_string('model_name', 'inception_v3.ckpt', 'name of checkpoint file')
flags.DEFINE_string('image_dir', 'image_dir', 'directory for storing adversarial examples')

############################
#   renderer setting       #
############################

flags.DEFINE_string('obj', '3d_model/barrel.obj', '.obj file path')
flags.DEFINE_string('texture', '3d_model/barrel.jpg', 'texture file path')

flags.DEFINE_float('camera_distance_min', 1.8, 'minimum camera distance')
flags.DEFINE_float('camera_distance_max', 2.3, 'maximum camera distance')

flags.DEFINE_float('x_translation_min', -0.05, 'minimum translation along x-axis')
flags.DEFINE_float('x_translation_max', 0.05, 'maximum translation along x-axis')

flags.DEFINE_float('y_translation_min', -0.05, 'minimum translation along y-axis')
flags.DEFINE_float('y_translation_max', 0.05, 'maximum translation along y-axis')

############################
# post-processing setting  #
############################

flags.DEFINE_boolean('print_error', True, 'consider printing error for textures')
flags.DEFINE_boolean('photo_error', True, 'consider photography error for images')

flags.DEFINE_float('background_min', 0.1, 'minimum rgb value for background')
flags.DEFINE_float('background_max', 1.0, 'maximum rgb value for background')

flags.DEFINE_float('light_add_min', -0.15, 'minimum additive lighten/darken')
flags.DEFINE_float('light_add_max', 0.15, 'maximum additive lighten/darken')

flags.DEFINE_float('light_mult_min', 0.5, 'minimum multiplicative lighten/darken')
flags.DEFINE_float('light_mult_max', 2.0, 'maximum multiplicative lighten/darken')

flags.DEFINE_float('channel_add_min', -0.15, 'minimum per channel additive lighten/darken')
flags.DEFINE_float('channel_add_max', 0.15, 'maximum per channel additive lighten/darken')

flags.DEFINE_float('channel_mult_min', 0.7, 'minimum per channel multiplicative lighten/darken')
flags.DEFINE_float('channel_mult_max', 1.3, 'maximum per channel multiplicative lighten/darken')

flags.DEFINE_float('stddev', 0.1, 'stddev for gaussian noise')

cfg = tf.compat.v1.flags.FLAGS
