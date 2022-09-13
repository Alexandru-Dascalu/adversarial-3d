import tensorflow as tf


flags = tf.compat.v1.flags

############################
#    hyper parameters      #
############################

BATCH_SIZE = 8
flags.DEFINE_float('batch_reuse_ratio', 0.8, 'percentage of batch samples that are reused in the next iteration')
flags.DEFINE_integer('iterations', 10000, 'iterations')
flags.DEFINE_float('loss_early_stopping_threshold', 0.5, 'threshold for loss to stop optimisation early')
flags.DEFINE_float('learning_rate', 0.003, 'initial learning rate')
flags.DEFINE_float('l2_weight', 0.025, 'the weighting factor for l2 loss')
# 463 - broom
TARGET_LABEL = 463
TRUE_LABELS = [427]

############################
#   environment setting    #
############################

flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_string('image_dir', 'adv_textures', 'directory for storing adversarial examples')

############################
#   renderer setting       #
############################

OBJ_PATH = 'dataset/barrel/barrel.obj'
TEXTURE_PATH = 'dataset/barrel/barrel.jpg'
NAME = "barrel"

flags.DEFINE_float('camera_distance_min', 1.8, 'minimum camera distance')
flags.DEFINE_float('camera_distance_max', 2.3, 'maximum camera distance')

flags.DEFINE_float('x_translation_min', -0.05, 'minimum translation along x-axis')
flags.DEFINE_float('x_translation_max', 0.05, 'maximum translation along x-axis')

flags.DEFINE_float('y_translation_min', -0.05, 'minimum translation along y-axis')
flags.DEFINE_float('y_translation_max', 0.05, 'maximum translation along y-axis')

############################
# post-processing setting  #
############################

flags.DEFINE_boolean('print_error', False, 'consider printing error for textures')
flags.DEFINE_boolean('photo_error', False, 'consider photography error for images')

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
