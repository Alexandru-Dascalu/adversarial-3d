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
        # with a batch size of 12, each of these take up around 13 MB once the images are actually rendered, if the
        # resolution is just 299x299
        self.std_images = None
        self.adv_images = None

        self.uv_mapping = tf.constant(0)
        self.top_k_predictions = []
        self.loss = 0

        self.optimiser = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        self.victim_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=True,
            weights='imagenet',
            classifier_activation=None
        )
        self.victim_model.trainable = False

    def optimisation_step(self, uv_mapping):
        """Perform one step of optmising the adversarial texture.

        Parameters
        ----------
        uv_mapping : numpy array
            A numpy array with shape [batch_size, image_height, image_width, 2]. Represents the UV mappings for each
            image in the batch. These mappigns are used to create the image from the textures.
        """
        self.uv_mapping = uv_mapping

        self.optimiser.minimize(self.loss_function, var_list=[self.adv_texture])
        # clip optimised texture to ensure its elements are between 0 and 1
        self.adv_texture.assign(tf.clip_by_value(self.adv_texture, 0, 1), name="clip optimised adv texture")

    def __call__(self):
        """Use UV mapping to create batch_seize images with both the normal and adversarial texture, then pass the
        adversarial images as input to the victim model to get logits. UV mapping is the matrix M used to transform
        texture x into the image with rendered object, as explained in the paper.

        Returns
        -------
        logits_v3
            Tensor of shape batch_size x 1000, representing the logits obtained by passing the adversarial images as
            input to the victim model.
        """
        # discard the images generated in the previous iteration
        self.std_images = None
        self.adv_images = None

        # create each image in batch from texture one at a time. We do this instead of all at once so that we need less
        # memory (a 12 x 2048 x 2048 x 3 tensor is 600 MB, and we would create multiple ones)
        for i in range(cfg.batch_size):
            self.create_image(i)

        # add background colour to rendered images.
        self.add_background()

        # check if we apply random noise to simulate camera noise
        if cfg.photo_error:
            self.apply_photo_error()

        # TODO: clip or scale to [0.0, 1.0]?
        self.std_images, self.adv_images = self.normalize(self.std_images, self.adv_images)

        # Pass images through trained model and get predictions in the form of logits
        # why scale the images?
        scaled_images = 2.0 * self.adv_images - 1.0
        logits_v3 = self.victim_model(scaled_images)

        return logits_v3

    def loss_function(self):
        """Perform one step of optmising the adversarial texture.

        Returns
        -------
        loss
            Tensor with element, representing the value of the loss function
        """
        prediction_logits = self()
        _, self.top_k_predictions = tf.nn.top_k(tf.nn.softmax(prediction_logits), k=5)

        # Calculate cross entropy loss for predictions
        labels = tf.constant(cfg.target, dtype=tf.int64, shape=[cfg.batch_size])
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=prediction_logits)
        # penalty term loss
        l2_loss = tf.reduce_sum(
            input_tensor=tf.square(tf.subtract(self.std_images, self.adv_images)), axis=[1, 2, 3])

        loss = cross_entropy_loss + cfg.l2_weight * l2_loss
        # reduce loss tensor to one scalar representing the average loss across the batch
        loss = tf.reduce_mean(loss)

        self.loss = loss
        return loss

    def get_diff(self):
        """Gets the difference vector between the adversarial texture and the original texture.

        Returns
        -------
        diff
            Numpy array.
        """
        diff_tensor = self.adv_texture - self.std_texture
        return diff_tensor.numpy()

    def create_image(self, index_in_batch):
        # check if we should add print errors, so that the adversarial texture may be used for a 3D printed object
        # and still be effective
        if cfg.print_error:
            std_texture, adv_texture = AdversarialNet.apply_print_error(self.std_texture, self.adv_texture)

        # Get UV map for this rendering in the batch. tfa.image.resampler requires the first dimension of UV map to be
        # batch size, so we add an extra dimension with one element
        image_uv_map = np.expand_dims(self.uv_mapping[index_in_batch], axis=0)

        # use UV mapping to create an images corresponding to an individual render by sampling from the texture
        # Resulting tensors are of shape 1 x image_width x image_height x 3
        std_image = tfa.image.resampler(std_texture, image_uv_map)
        adv_image = tfa.image.resampler(adv_texture, image_uv_map)

        # the two tensors are 4D, with only one 3D tensor in the first dimension
        if self.std_images is None:
            self.std_images = std_image
            self.adv_images = adv_image
        else:
            self.std_images = tf.concat([self.std_images, std_image], axis=0)
            self.adv_images = tf.concat([self.adv_images, adv_image], axis=0)

    @staticmethod
    def apply_print_error(std_texture, adv_texture):
        multiplier = tf.random.uniform(
            [1, 1, 1, 3],
            cfg.channel_mult_min,
            cfg.channel_mult_max
        )
        addend = tf.random.uniform(
            [1, 1, 1, 3],
            cfg.channel_add_min,
            cfg.channel_add_max
        )
        std_texture = AdversarialNet.transform(std_texture, multiplier, addend)
        adv_texture = AdversarialNet.transform(adv_texture, multiplier, addend)

        return std_texture, adv_texture

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

        Parameters
        ----------
            x : 3-D tensor with shape [height, width, 3]
                The image to be repeated.
            times : int
                How many times to repeat the 3D tensor in the first dimension.
        Returns
        -------
            A 4-D tensor with shape [times, height, size, 3]
        """
        return tf.tile(tf.expand_dims(x, 0), [times, 1, 1, 1])

    @staticmethod
    def transform(x, a, b):
        """Apply transform a * x + b element-wise.

         Parameters
        ----------
            x : tensor
            a : tensor
            b : tensor
        """
        return tf.add(tf.multiply(a, x), b)

    def add_background(self):
        """Colours the background pixels of the image with a random colour.
        """
        # compute a mask with True values for each pixel which represents the object, and False for background pixels.
        mask = tf.reduce_all(
            input_tensor=tf.not_equal(self.uv_mapping, 0.0), axis=3, keepdims=True)
        # generate random background colour for each image in batch
        color = tf.random.uniform(
            [cfg.batch_size, 1, 1, 3], cfg.background_min, cfg.background_max)

        self.std_images = AdversarialNet.set_background(self.std_images, mask, color)
        self.adv_images = AdversarialNet.set_background(self.adv_images, mask, color)

    @staticmethod
    def set_background(x, mask, colours):
        """Sets background color of an image according to a boolean mask.
        
        Parameters
        ----------
            x: A 4-D tensor with shape [batch_size, height, size, 3]
                The images to which a background will be added.
            mask: boolean mask with shape [batch_size, height, width, 1]
                The mask used for determining where are the background pixels. Has False for background pixels, 
                True otherwise.
            colours: tensor with shape [batch_size, 1, 1, 3].
                The background colours for each image
        """
        mask = tf.tile(mask, [1, 1, 1, 3])
        inverse_mask = tf.logical_not(mask)

        return tf.cast(mask, tf.float32) * x + tf.cast(inverse_mask, tf.float32) * colours

    @staticmethod
    def normalize(x, y):
        minimum = tf.minimum(tf.reduce_min(input_tensor=x, axis=[1, 2], keepdims=True),
                             tf.reduce_min(input_tensor=y, axis=[1, 2, 3], keepdims=True))
        maximum = tf.maximum(tf.reduce_max(input_tensor=x, axis=[1, 2], keepdims=True),
                             tf.reduce_max(input_tensor=y, axis=[1, 2, 3], keepdims=True))

        minimum = tf.minimum(minimum, 0)
        maximum = tf.maximum(maximum, 1)

        return (x - minimum) / (maximum - minimum), (y - minimum) / (maximum - minimum)
