import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from config import cfg


class AdversarialNet(tf.Module):

    def __init__(self, texture):
        """Construct a model for generating 3D adversarial examples by optimising the texture of the object to be
        adversarial under a variety of random rotations, translations, camera and printing errors.

        Parameters
        ----------
        texture : numpy array
            A numpy array with shape [height, width, 3]
        """
        super(AdversarialNet, self).__init__()

        # 2048x2048x3 tensors, each takes 50 MB
        self.std_texture = tf.constant(texture, name='texture')
        self.adv_texture = tf.Variable(self.std_texture, trainable=True, name='adv_texture')

        # Initialise tensors that will hold the current batch of rendered images with normal and adversarial textures
        batch_tensor_size = (cfg.batch_size,) + self.std_texture.shape
        self.std_images = np.zeros(batch_tensor_size, dtype=np.float32)
        self.adv_images = np.zeros(batch_tensor_size, dtype=np.float32)
        self.uv_mapping = []

        self.top_k_predictions = []
        self.loss = 0

        self.optimiser = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        self.victim_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=True,
            weights='imagenet',
            classifier_activation=None
        )
        self.victim_model.trainable = False

    # UV mapping is the matrix M used to transform texture x into the image with rendered object. Since UV mapping
    # is calculated for each individual 3D render, we just initialise it with random values.
    def __call__(self):
        # self.uv_mapping = tf.Variable(np.empty((cfg.batch_size, 0, 0, 2), dtype=np.float32),
        #                               shape=[cfg.batch_size, None, None, 2], name='uv_mapping')

        # replicate textures for each adv example in batch
        std_textures = AdversarialNet.repeat(self.std_texture, cfg.batch_size)
        adv_textures = AdversarialNet.repeat(self.adv_texture, cfg.batch_size)

        if cfg.print_error:
            std_textures, adv_textures = AdversarialNet.apply_print_error(std_textures, adv_textures)

        # use UV mapping to create images of different rendered objects by sampling from the texture
        self.std_images = tfa.image.resampler(std_textures, self.uv_mapping)
        self.adv_images = tfa.image.resampler(adv_textures, self.uv_mapping)

        # add background colour to rendered images.
        self.add_background()

        if cfg.photo_error:
            self.apply_photo_error()

        # TODO: clip or scale to [0.0, 1.0]?
        # std_images = tf.clip_by_value(std_images, 0, 1)
        # adv_images = tf.clip_by_value(adv_images, 0, 1)
        self.std_images, self.adv_images = self.normalize(self.std_images, self.adv_images)

        # Pass images through trained model and get predictions in the form of logits
        # why scale the images?
        scaled_images = 2.0 * self.adv_images - 1.0
        logits_v3 = self.victim_model(scaled_images)

        return logits_v3

    def loss_function(self):
        prediction_logits = self()
        _, self.top_k_predictions = tf.nn.top_k(tf.nn.softmax(prediction_logits), k=5)

        # Calculate cross entropy loss for predictions
        labels = tf.constant(cfg.target, dtype=tf.int64, shape=[cfg.batch_size])
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=prediction_logits)
        l2_loss = tf.reduce_sum(
            input_tensor=tf.square(tf.subtract(self.std_images, self.adv_images)), axis=[1, 2, 3])

        # loss = tf.reduce_mean(input_tensor=cross_entropy_loss + cfg.l2_weight * l2_loss)
        loss = cross_entropy_loss + cfg.l2_weight * l2_loss
        loss = tf.reduce_mean(loss)
        self.loss = loss
        return loss

    def optimisation_step(self, uv_mapping):
        self.uv_mapping = uv_mapping

        self.optimiser.minimize(self.loss_function, var_list=[self.adv_texture])
        self.adv_texture.assign(tf.clip_by_value(self.adv_texture, 0, 1), name="clip optimised adv texture")

    def get_diff(self):
        diff_tensor = self.adv_texture - self.std_texture
        return diff_tensor.numpy()

    @staticmethod
    def apply_print_error(std_textures, adv_textures):
        multiplier = tf.random.uniform(
            [cfg.batch_size, 1, 1, 3],
            cfg.channel_mult_min,
            cfg.channel_mult_max
        )
        addend = tf.random.uniform(
            [cfg.batch_size, 1, 1, 3],
            cfg.channel_add_min,
            cfg.channel_add_max
        )
        std_textures = AdversarialNet.transform(std_textures, multiplier, addend)
        adv_textures = AdversarialNet.transform(adv_textures, multiplier, addend)

        return std_textures, adv_textures

    def apply_photo_error(self):
        multiplier = tf.random.uniform(
            [cfg.batch_size, 1, 1, 1],
            cfg.light_mult_min,
            cfg.light_mult_max
        )
        addend = tf.random.uniform(
            [cfg.batch_size, 1, 1, 1],
            cfg.light_add_min,
            cfg.light_add_max
        )
        self.std_images = AdversarialNet.transform(self.std_images, multiplier, addend)
        self.adv_images = AdversarialNet.transform(self.adv_images, multiplier, addend)

        gaussian_noise = tf.random.truncated_normal(
            tf.shape(input=self.std_images),
            stddev=tf.random.uniform([1], maxval=cfg.stddev)
        )

        self.std_images += gaussian_noise
        self.adv_images += gaussian_noise

    @staticmethod
    def repeat(x, times):
        """Repeat a image multiple times to generate a batch
        Args:
            x: A 3-D tensor with shape [height, width, 3]
            times: How many times to repeat the 3D tensor in the first dimension,
        Returns:
            A 4-D tensor with shape [times, height, size, 3]
        """
        return tf.tile(tf.expand_dims(x, 0), [times, 1, 1, 1])

    @staticmethod
    def transform(x, a, b):
        """Apply transform a * x + b element-wise
        """
        return tf.add(tf.multiply(a, x), b)

    def add_background(self):
        mask = tf.reduce_all(
            input_tensor=tf.not_equal(self.uv_mapping, 0.0), axis=3, keepdims=True)
        color = tf.random.uniform(
            [cfg.batch_size, 1, 1, 3], cfg.background_min, cfg.background_max)

        self.std_images = AdversarialNet.set_backgroud(self.std_images, mask, color)
        self.adv_images = AdversarialNet.set_backgroud(self.adv_images, mask, color)

    @staticmethod
    def set_backgroud(x, mask, color):
        """Set background color according to a boolean mask
        Args:
            x: A 4-D tensor with shape [batch_size, height, size, 3]
            mask: boolean mask with shape [batch_size, height, width, 1]
            color: background color with shape [batch_size, 1, 1, 3]
        """
        mask = tf.tile(mask, [1, 1, 1, 3])
        inv = tf.logical_not(mask)

        return tf.cast(mask, tf.float32) * x + tf.cast(inv, tf.float32) * color

    @staticmethod
    def normalize(x, y):
        minimum = tf.minimum(tf.reduce_min(input_tensor=x, axis=[1, 2], keepdims=True),
                             tf.reduce_min(input_tensor=y, axis=[1, 2, 3], keepdims=True))
        maximum = tf.maximum(tf.reduce_max(input_tensor=x, axis=[1, 2], keepdims=True),
                             tf.reduce_max(input_tensor=y, axis=[1, 2, 3], keepdims=True))

        minimum = tf.minimum(minimum, 0)
        maximum = tf.maximum(maximum, 1)

        return (x - minimum) / (maximum - minimum), (y - minimum) / (maximum - minimum)
