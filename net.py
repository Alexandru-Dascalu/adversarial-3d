import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import skimage.color

from config import cfg
import config


class AdversarialNet(tf.Module):

    def __init__(self, texture):
        """
        Construct a model for generating 3D adversarial examples by optimising the texture of the object to be
        adversarial under a variety of random rotations, translations, camera and printing errors.

        Parameters
        ----------
        texture : numpy array
            A numpy array with shape [height, width, 3]
        """
        super(AdversarialNet, self).__init__()

        # texture_width x texture_width x 3 tensors.
        self.std_texture = tf.constant(texture, name='texture')
        self.adv_texture = tf.Variable(self.std_texture, trainable=True, name='adv_texture')

        # Initialise tensors that will hold the current batch of rendered images with normal and adversarial textures.
        # With a batch size of 12, each of these take up around 13 MB once the images are actually rendered, if the
        # resolution is just 299x299
        batch_tensor_size = (config.BATCH_SIZE, 299, 299, 3)
        self.std_images = tf.zeros(batch_tensor_size, dtype=tf.dtypes.float32)
        self.adv_images = tf.zeros(batch_tensor_size, dtype=tf.dtypes.float32)

        # uv maps used to make renders
        self.uv_mapping = tf.zeros((1,))
        # logits of te neural network being attacked
        self.logits = tf.zeros((config.BATCH_SIZE, 1000))
        self.top_k_predictions = []

        self.main_loss_history = []
        self.l2_loss_history = []
        self.tfr_history = []

        self.optimiser = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        self.victim_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=True,
            weights='imagenet',
            classifier_activation=None
        )
        self.victim_model.trainable = False

    def optimisation_step(self, uv_mapping):
        """
        Perform one step of optimising the adversarial texture.

        Parameters
        ----------
        uv_mapping : numpy array
            A numpy array with shape [batch_size, image_height, image_width, 2]. Represents the UV mappings for each
            image in the batch. These mappings are used to create the image from the textures.
        """
        self.uv_mapping = uv_mapping

        self.optimiser.minimize(self.loss_function, var_list=[self.adv_texture])
        # clip optimised texture to ensure its elements are between 0 and 1
        self.adv_texture.assign(tf.clip_by_value(self.adv_texture, 0, 1), name="clip optimised adv texture")

    def loss_function(self):
        """
        The loss function for creating the adversarial texture. We use gradient descent optimisation to minimise it.

        Returns
        -------
        loss
            Tensor with one element, representing the value of the loss function
        """
        self()
        _, self.top_k_predictions = tf.nn.top_k(tf.nn.softmax(self.logits), k=5)

        # Calculate cross entropy loss for predictions
        labels = tf.constant(config.TARGET_LABEL, dtype=tf.int64, shape=[config.BATCH_SIZE])
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.logits)

        lab_std_images = tf.convert_to_tensor(AdversarialNet.get_normalised_lab_image(self.std_images.numpy()))
        lab_adv_images = tf.convert_to_tensor(AdversarialNet.get_normalised_lab_image(self.adv_images.numpy()))

        # penalty term loss
        l2_loss = tf.sqrt(tf.reduce_sum(
            input_tensor=tf.square(tf.subtract(lab_std_images, lab_adv_images)), axis=[1, 2, 3]))

        # reduce loss tensors to one scalar representing the average loss across the batch
        cross_entropy_loss = tf.reduce_mean(cross_entropy_loss)
        l2_loss = tf.reduce_mean(l2_loss)

        # add losses for this optimisation step to history
        self.main_loss_history.append(cross_entropy_loss.numpy())
        self.l2_loss_history.append(cfg.l2_weight * l2_loss.numpy())
        self.tfr_history.append(self.get_tfr())

        loss = cross_entropy_loss + cfg.l2_weight * l2_loss
        return loss

    def __call__(self):
        """
        Uses UV mapping to create batch_seize images with both the normal and adversarial texture, then pass the
        adversarial images as input to the victim model to get logits. UV mapping is the matrix M used to transform
        texture x into the image with rendered object, as explained in the paper.

        Returns
        -------
        Tensor of shape batch_size x 1000, representing the logits obtained by passing the adversarial images as
        input to the victim model.
        """
        # for the first iteration, we have a whole new batch of renders, for other iterations we have only around 20%
        # new renders
        num_new_renders = self.uv_mapping.shape[0]

        # create each image in batch from texture one at a time. We do this instead of all at once so that we need less
        # memory (a 12 x 2048 x 2048 x 3 tensor is 600 MB, and we would create multiple ones). We make the first image
        # outside of the loop to initialise the list of new images, and to avoid putting an if statement in the loop
        new_std_images, new_adv_images = self.create_image(self.uv_mapping[0])
        for i in range(1, num_new_renders):
            std_image, adv_image = self.create_image(self.uv_mapping[i])
            new_std_images = tf.concat([new_std_images, std_image], axis=0)
            new_adv_images = tf.concat([new_adv_images, adv_image], axis=0)

        # add background colour to rendered images.
        new_std_images, new_adv_images = self.add_background(new_std_images, new_adv_images)

        # check if we apply random noise to simulate camera noise
        if cfg.photo_error:
            new_std_images, new_adv_images = AdversarialNet.apply_photo_error(self.uv_mapping.shape[0], new_std_images,
                                                                              new_adv_images)

        new_std_images, new_adv_images = AdversarialNet.normalize(new_std_images, new_adv_images)
        # Pass images through trained model and get predictions in the form of logits.
        # we scale the images because inceptionv3 requires images with values between -1 and 1
        new_logits = self.victim_model(2.0 * new_adv_images - 1.0)

        # add images and prediction logits for the new renders to the whole batch
        self.std_images = AdversarialNet.insert_new_elements_in_tensor(self.std_images, new_std_images)
        self.adv_images = AdversarialNet.insert_new_elements_in_tensor(self.adv_images, new_adv_images)
        self.logits = AdversarialNet.insert_new_elements_in_tensor(self.logits, new_logits)

    def create_image(self, uv_map):
        """
        Create standard and adversarial images from the respective textures using the given UV map.

        Parameters
        ----------
        uv_map : numpy array
            A numpy array with shape [image_height, image_width, 2]. Represents the UV mappings for an
            image in the batch. This mappign is used to create the images from the textures.

        Returns
        -------
        tuple
            Two tensors. The first one is of shape num_new_renders x 299 x 299 x 3, representing the images of the new
            renders with the normal texture. The second is of shape num_new_renders x 299 x 299 x 3, representing the
            images of the new renders with the adversarial texture.
        """
        # check if we should add print errors, so that the adversarial texture may be used for a 3D printed object
        # and still be effective
        if cfg.print_error:
            std_texture, adv_texture = AdversarialNet.apply_print_error(self.std_texture, self.adv_texture)
        else:
            std_texture = self.std_texture
            adv_texture = self.adv_texture

        # tfa.resampler requires input to be in shape batch_size x height x width x channels, so we insert a new
        # dimension
        std_texture = tf.expand_dims(std_texture, axis=0)
        adv_texture = tf.expand_dims(adv_texture, axis=0)
        uv_map = np.expand_dims(uv_map, axis=0)

        # use UV mapping to create an images corresponding to an individual render by sampling from the texture
        # Resulting tensors are of shape 1 x image_width x image_height x 3
        std_image = tfa.image.resampler(std_texture, uv_map)
        adv_image = tfa.image.resampler(adv_texture, uv_map)

        return std_image, adv_image

    def add_background(self, new_std_images, new_adv_images):
        """
        Colours the background pixels of the image with a random colour. Each image with the normal texture will have
        the same background as its adversarial counterpart.

        Parameters
        ----------
        new_std_images : tensor
            A tensor with shape [batch_size, image_height, image_width, 3]. Represents the new rendered images with the
            normal texture.
        new_adv_images : tensor
            A tensor with shape [batch_size, image_height, image_width, 3]. Represents the new rendered images with the
            adversarial texture.

        Returns
        -------
        tuple
            Two tensors. The first one is of shape num_new_renders x 299 x 299 x 3, representing the images of the new
            renders with the normal texture. The second is of shape num_new_renders x 299 x 299 x 3, representing the
            images of the new renders with the adversarial texture. Both batches have coloured backgrounds now.
        """
        # compute a mask with True values for each pixel which represents the object, and False for background pixels.
        mask = tf.reduce_all(tf.not_equal(self.uv_mapping, 0.0), axis=3, keepdims=True)
        # generate random background colour for each image in batch
        num_new_renders = new_std_images.shape[0]
        color = tf.random.uniform(
            [num_new_renders, 1, 1, 3], cfg.background_min, cfg.background_max)

        new_std_images = AdversarialNet.set_background(new_std_images, mask, color)
        new_adv_images = AdversarialNet.set_background(new_adv_images, mask, color)

        return new_std_images, new_adv_images

    @staticmethod
    def set_background(images, mask, colours):
        """
        Sets background color of an image according to a boolean mask.

        Parameters
        ----------
        images: tensor
            The images to which a background will be added. A 4-D tensor with shape [batch_size, height, width, 3].
        mask: tensor
            The mask used for determining where are the background pixels. Has False for background pixels, True otherwise.
            Has shape [batch_size, height, width, 1].
        colours: tensor with shape [batch_size, 1, 1, 3].
            The background colours for each image. Has shape [batch_size, 1, 1, 3].
        """
        mask = tf.tile(mask, [1, 1, 1, 3])
        inverse_mask = tf.logical_not(mask)

        return tf.cast(mask, tf.float32) * images + tf.cast(inverse_mask, tf.float32) * colours

    @staticmethod
    def apply_print_error(std_texture, adv_texture):
        """
        Applies print error to a pair of textures. It will linearly scale each colour channel independently. Both the
        standard and adversarial textures will be scaled using the exact same random parameters.

        Parameters
        ----------
        std_texture : tensor
            The normal texture of the object. A 3-D tensor with shape [texture_height, texture_width, 3].
        adv_texture : tensor
            The adversarial texture of the object. A 3-D tensor with shape [texture_height, texture_width, 3].

        Returns
        -------
        tuple
            A tuple with two tensors. The first is the normal texture with the print error scaling applied, the second
            is the adversarial texture with the same scaling applied to it.
        """
        multiplier = tf.random.uniform(
            [1, 1, 3],
            cfg.channel_mult_min,
            cfg.channel_mult_max
        )
        addend = tf.random.uniform(
            [1, 1, 3],
            cfg.channel_add_min,
            cfg.channel_add_max
        )
        std_texture = AdversarialNet.transform(std_texture, multiplier, addend)
        adv_texture = AdversarialNet.transform(adv_texture, multiplier, addend)

        return std_texture, adv_texture

    @staticmethod
    def apply_photo_error(num_new_renders, std_images, adv_images):
        """
        Applies photo error to a pair of images. It will linearly scale each image to lighten or darken it, then add
        gaussian noise to simulate camera noise. Both the standard and adversarial textures will be scaled using the
        exact same random parameters.

        Parameters
        ----------
        num_new_renders : int
            Number of new renders in the current step.
        std_images : tensor
            The images with normal texture of the object. A 4-D tensor with shape
            [batch_size, texture_height, texture_width, 3].
        adv_images : tensor
            The images with normal texture of the object. A 4-D tensor with shape
            [batch_size, texture_height, texture_width, 3].

        Returns
        -------
        tuple
            Two tensors. The first one is of shape num_new_renders x 299 x 299 x 3, representing the images of the new
            renders with the normal texture. The second is of shape num_new_renders x 299 x 299 x 3, representing the
            images of the new renders with the adversarial texture. Both batches have had lighting and camera noise
            added to them.
        """
        multiplier = tf.random.uniform(
            [num_new_renders, 1, 1, 1],
            cfg.light_mult_min,
            cfg.light_mult_max
        )
        addend = tf.random.uniform(
            [num_new_renders, 1, 1, 1],
            cfg.light_add_min,
            cfg.light_add_max
        )
        std_images = AdversarialNet.transform(std_images, multiplier, addend)
        adv_images = AdversarialNet.transform(adv_images, multiplier, addend)

        gaussian_noise = tf.random.truncated_normal(
            shape=tf.shape(std_images),
            stddev=tf.random.uniform([1], maxval=cfg.stddev)
        )

        std_images += gaussian_noise
        adv_images += gaussian_noise

        return std_images, adv_images

    @staticmethod
    def insert_new_elements_in_tensor(old_data, new_elements_tensor):
        """
        Insert n new elements at the end of a tensor. It shifts last (tensor_size - n) elements to the left, thus
        preserving them and over-writing the first n elements. Then it will put the n new elements at the end.

        Parameters
        ----------
        old_data : tensor
            The tensor with the original data.
        new_elements_tensor : tensor
            A tensor with the new elements to be added.
        """
        num_new_elements = new_elements_tensor.shape[0]
        # get a list of tensors, each being the tensor in the ith row of the input tensor x
        tensor_list = tf.unstack(old_data)

        # shift elements that will be re-used to the left
        tensor_list[:-num_new_elements] = tensor_list[num_new_elements:]
        # place new elements at the end of the list
        tensor_list[-num_new_elements:] = new_elements_tensor

        return tf.stack(tensor_list)

    @staticmethod
    def repeat(image, times):
        """
        Repeat an image multiple times to generate a batch

        Parameters
        ----------
        image : tensor
            A 3-D tensor with shape [height, width, 3]. It is the image to be repeated.
        times : int
            How many times to repeat the 3D tensor in the first dimension.
        Returns
        -------
        tensor
            A 4-D tensor with shape [times, height, size, 3]
        """
        return tf.tile(tf.expand_dims(image, 0), [times, 1, 1, 1])

    @staticmethod
    def transform(x, a, b):
        """
        Apply transform a * x + b element-wise.

        Parameters
        ----------
        x : tensor
            A tensor to transform.
        a : tensor
            The multiplier tensor.
        b : tensor
            The addend tensor.
        """
        return tf.add(tf.multiply(a, x), b)

    @staticmethod
    def normalize(std_images, adv_images):
        """
        Normalises rendered images of the object, as after the rendering process, pixel values may be outside [0, 1].
        This method performs linear normalisation, though if an image does not have invalid values, it will not be scaled.
        Moreover, each normal image will use the same scale for normalisation as its adversarial counterpart, because each
        pair of a normal and an adversarial image must be identical, save for the adversarial noise on the texture itself.

        Parameters
        ----------
        std_images : tensor
            The images with normal texture of the object. A 4-D tensor with shape
            [batch_size, texture_height, texture_width, 3].
        adv_images : tensor
            The images with normal texture of the object. A 4-D tensor with shape
            [batch_size, texture_height, texture_width, 3].

        Returns
        -------
        tuple
            Two tensors. The first one is of shape batch_size x 299 x 299 x 3, representing the normalised images with
            the normal texture. The second is of shape batch size x 299 x 299 x 3, representing the normalised
            images with the adversarial texture.
        """
        std_images_minimums = tf.reduce_min(std_images, axis=[1, 2, 3], keepdims=True)
        adv_images_minimums = tf.reduce_min(adv_images, axis=[1, 2, 3], keepdims=True)

        std_images_maximums = tf.reduce_max(std_images, axis=[1, 2, 3], keepdims=True)
        adv_images_maximums = tf.reduce_max(adv_images, axis=[1, 2, 3], keepdims=True)

        minimum = tf.minimum(std_images_minimums, adv_images_minimums)
        maximum = tf.maximum(std_images_maximums, adv_images_maximums)

        minimum = tf.minimum(minimum, 0)
        maximum = tf.maximum(maximum, 1)

        return (std_images - minimum) / (maximum - minimum), (adv_images - minimum) / (maximum - minimum)

    @staticmethod
    def get_normalised_lab_image(rgb_images):
        """
        Turn a numpy array representing normalised RGB images into equivalent normalised images in the LAB
        colour space.

        Parameters
        ----------
        rgb_images : numpy array of size batch_size x 299 x 299 x 3
            The image which we want to convert to LAB space. Each value in it must be between 0 and 1. Is a numpy array
            of size batch_size x 299 x 299 x 3 .
        Returns
        -------
        numpy array
            A 4-D numpy array with shape [batch_size, 299, 299, 3] and with values between 0 and 1.
        """
        assert rgb_images.shape[1] == 299
        assert rgb_images.shape[2] == 299
        assert rgb_images.shape[3] == 3

        lab_images = skimage.color.rgb2lab(rgb_images)

        # normalise the lightness channel, which has values between 0 and 100
        lab_images[..., 0] = lab_images[..., 0] / 100
        # normalise the greenness-redness and blueness-yellowness channels, which normally are between -128 and 127
        lab_images[..., 1] = (lab_images[..., 1] + 128) / 255
        lab_images[..., 2] = (lab_images[..., 2] + 128) / 255

        return lab_images

    def get_diff(self):
        """Gets the difference vector between the adversarial texture and the original texture.

        Returns
        -------
        numpy array
            Numpy array representing the difference vector.
        """
        diff_tensor = self.adv_texture - self.std_texture
        return diff_tensor.numpy()

    def get_tfr(self):
        """
        Calculates TFR for the victim model predictions on the current batch of rendered images.

        Returns
        -------
        int
            The TFR.
        """
        target_reached = 0
        # iterate through the predicted label for each image
        for top_prediction in self.top_k_predictions[:, 0]:
            if top_prediction == config.TARGET_LABEL:
                target_reached += 1

        tfr = target_reached / config.BATCH_SIZE
        return tfr
