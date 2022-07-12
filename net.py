import tensorflow as tf
import tensorflow_addons as tfa

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
        batch_tensor_size = (cfg.batch_size, 299, 299, 3)
        # with a batch size of 12, each of these take up around 600 MB
        self.std_images = tf.zeros(batch_tensor_size, dtype=tf.dtypes.float32)
        self.adv_images = tf.zeros(batch_tensor_size, dtype=tf.dtypes.float32)

        self.uv_mapping = tf.zeros((1,))
        self.logits = tf.zeros((cfg.batch_size, 1000))
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

    def loss_function(self):
        """Perform one step of optmising the adversarial texture.

        Returns
        -------
        loss
            Tensor with element, representing the value of the loss function
        """
        self()
        _, self.top_k_predictions = tf.nn.top_k(tf.nn.softmax(self.logits), k=5)

        # Calculate cross entropy loss for predictions
        labels = tf.constant(cfg.target, dtype=tf.int64, shape=[cfg.batch_size])
        cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=self.logits)
        # penalty term loss
        l2_loss = tf.reduce_sum(
            input_tensor=tf.square(tf.subtract(self.std_images, self.adv_images)), axis=[1, 2, 3])

        loss = cross_entropy_loss + cfg.l2_weight * l2_loss
        # reduce loss tensor to one scalar representing the average loss across the batch
        loss = tf.reduce_mean(loss)

        self.loss = loss
        return loss

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
        # for the first iteration, we have a whole new batch of renders, for other iterations we have only around 20%
        # new renders
        new_std_images, new_adv_images = self.create_images_from_texture()

        # add background colour to rendered images.
        new_std_images, new_adv_images = self.add_background(new_std_images, new_adv_images)

        # check if we apply random noise to simulate camera noise
        if cfg.photo_error:
            new_std_images, new_adv_images = AdversarialNet.apply_photo_error(self.uv_mapping.shape[0], new_std_images,
                                                                              new_adv_images)

        # TODO: clip or scale to [0.0, 1.0]?
        new_std_images, new_adv_images = AdversarialNet.normalize(new_std_images, new_adv_images)
        # Pass images through trained model and get predictions in the form of logits
        # why scale the images?
        scaled_images = 2.0 * new_adv_images - 1.0
        new_logits = self.victim_model(scaled_images)

        # add images and prediction logits for the new renders to the whole batch
        self.std_images = AdversarialNet.insert_new_elements_in_tensor(self.std_images, new_std_images)
        self.adv_images = AdversarialNet.insert_new_elements_in_tensor(self.adv_images, new_adv_images)
        self.logits = AdversarialNet.insert_new_elements_in_tensor(self.logits, new_logits)

    def create_images_from_texture(self):
        """Create standard and adversarial images from the respective textures using the UV mappings of the new renders.

        Returns
        -------
        new_std_images
            Tensor of shape num_new_renders x 299 x 299 x 3, representing the images of the new renders with the normal
            texture.
        new_adv_images
            Tensor of shape num_new_renders x 299 x 299 x 3, representing the images of the new renders with the
            adversarial texture.
        """
        num_new_renders = self.uv_mapping.shape[0]
        # replicate textures for each adv example in batch
        std_textures = AdversarialNet.repeat(self.std_texture, num_new_renders)
        adv_textures = AdversarialNet.repeat(self.adv_texture, num_new_renders)

        # check if we should add print errors, so that the adversarial texture may be used for a 3D printed object and
        # still be effective
        if cfg.print_error:
            std_textures, adv_textures = AdversarialNet.apply_print_error(num_new_renders, std_textures, adv_textures)

        # use UV mapping to create images of different rendered objects by sampling from the texture
        new_std_images = tfa.image.resampler(std_textures, self.uv_mapping)
        new_adv_images = tfa.image.resampler(adv_textures, self.uv_mapping)

        return new_std_images, new_adv_images

    def add_background(self, new_std_images, new_adv_images):
        """Colours the background pixels of the image with a random colour.
        """
        # compute a mask with True values for each pixel which represents the object, and False for background pixels.
        mask = tf.reduce_all(
            input_tensor=tf.not_equal(self.uv_mapping, 0.0), axis=3, keepdims=True)
        # generate random background colour for each image in batch
        num_new_renders = new_std_images.shape[0]
        color = tf.random.uniform(
            [num_new_renders, 1, 1, 3], cfg.background_min, cfg.background_max)

        new_std_images = AdversarialNet.set_background(new_std_images, mask, color)
        new_adv_images = AdversarialNet.set_background(new_adv_images, mask, color)

        return new_std_images, new_adv_images

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
    def apply_print_error(num_new_renders, std_textures, adv_textures):
        multiplier = tf.random.uniform(
            [num_new_renders, 1, 1, 3],
            cfg.channel_mult_min,
            cfg.channel_mult_max
        )
        addend = tf.random.uniform(
            [num_new_renders, 1, 1, 3],
            cfg.channel_add_min,
            cfg.channel_add_max
        )
        std_textures = AdversarialNet.transform(std_textures, multiplier, addend)
        adv_textures = AdversarialNet.transform(adv_textures, multiplier, addend)

        return std_textures, adv_textures

    @staticmethod
    def apply_photo_error(num_new_renders, std_images, adv_images):
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
            tf.shape(input=std_images),
            stddev=tf.random.uniform([1], maxval=cfg.stddev)
        )

        std_images += gaussian_noise
        adv_images += gaussian_noise

        return std_images, adv_images

    @staticmethod
    def insert_new_elements_in_tensor(x, new_elements_tensor):
        """Insert n new elements at the end of a tensor. It shifts last (tensor_size - n) elements to the left, thus
        preserving them and over-writing the first n elements. Then it will put the n new elements at the end.

        Parameters
        ----------
        x : tensor
            The tensor with the original data.
        new_elements_tensor : tensor
            A tensor with the new elements to be added.
        """
        num_elements = new_elements_tensor.shape[0]
        # get a list of tensors, each being the tensor in the ith row of the input tensor x
        tensor_list = tf.unstack(x)

        # shift elements that will be re-used to the left
        tensor_list[:-num_elements] = tensor_list[num_elements:]
        # place new elements at the end of the list
        tensor_list[-num_elements:] = new_elements_tensor

        return tf.stack(tensor_list)

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

    @staticmethod
    def normalize(x, y):
        minimum = tf.minimum(tf.reduce_min(input_tensor=x, axis=[1, 2], keepdims=True),
                             tf.reduce_min(input_tensor=y, axis=[1, 2, 3], keepdims=True))
        maximum = tf.maximum(tf.reduce_max(input_tensor=x, axis=[1, 2], keepdims=True),
                             tf.reduce_max(input_tensor=y, axis=[1, 2, 3], keepdims=True))

        minimum = tf.minimum(minimum, 0)
        maximum = tf.maximum(maximum, 1)

        return (x - minimum) / (maximum - minimum), (y - minimum) / (maximum - minimum)

    def get_diff(self):
        """Gets the difference vector between the adversarial texture and the original texture.

        Returns
        -------
        diff
            Numpy array.
        """
        diff_tensor = self.adv_texture - self.std_texture
        return diff_tensor.numpy()
