import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from config import cfg


def render(std_texture, adv_texture, uv_maps):
    """Use UV mapping to create batch_seize images with both the normal and adversarial texture, then pass the
    adversarial images as input to the victim model to get logits. UV mapping is the matrix M used to transform
    texture x into the image with rendered object, as explained in the paper.

    Returns
    -------
    Tensor of shape batch_size x 1000, representing the logits obtained by passing the adversarial images as
    input to the victim model.
    """
    # create each image in batch from texture one at a time. We do this instead of all at once so that we need less
    # memory (a 12 x 2048 x 2048 x 3 tensor is 600 MB, and we would create multiple ones). We make the first image
    # outside of the loop to initialise the list of new images, and to avoid putting an if statement in the loop
    new_std_images, new_adv_images = create_image(std_texture, adv_texture, uv_maps[0])
    for i in range(1, uv_maps.shape[0]):
        std_image, adv_image = create_image(std_texture, adv_texture, uv_maps[i])
        new_std_images = tf.concat([new_std_images, std_image], axis=0)
        new_adv_images = tf.concat([new_adv_images, adv_image], axis=0)

    # add background colour to rendered images.
    new_std_images, new_adv_images = add_background(new_std_images, new_adv_images, uv_maps)

    # check if we apply random noise to simulate camera noise
    if cfg.photo_error:
        new_std_images, new_adv_images = apply_photo_error(uv_maps.shape[0], new_std_images, new_adv_images)

    new_std_images, new_adv_images = normalize(new_std_images, new_adv_images)
    return new_std_images, new_adv_images


def create_image(std_texture, adv_texture, uv_maps):
    """Create standard and adversarial images from the respective textures using the given UV mapping.

    Parameters
    ----------
    uv_maps : numpy array
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
        std_texture, adv_texture = apply_print_error(std_texture, adv_texture)

    # tfa.resampler requires input to be in shape batch_size x height x width x channels, so we insert a new
    # dimension
    std_texture = tf.expand_dims(std_texture, axis=0)
    adv_texture = tf.expand_dims(adv_texture, axis=0)
    uv_maps = np.expand_dims(uv_maps, axis=0)

    # use UV mapping to create an images corresponding to an individual render by sampling from the texture
    # Resulting tensors are of shape 1 x image_width x image_height x 3
    std_image = tfa.image.resampler(std_texture, uv_maps)
    adv_image = tfa.image.resampler(adv_texture, uv_maps)

    return std_image, adv_image


def add_background(new_std_images, new_adv_images, uv_maps):
    """Colours the background pixels of the image with a random colour.
    """
    # compute a mask with True values for each pixel which represents the object, and False for background pixels.
    mask = tf.reduce_all(tf.not_equal(uv_maps, 0.0), axis=3, keepdims=True)
    # generate random background colour for each image in batch
    num_new_renders = uv_maps.shape[0]
    color = tf.random.uniform(
        [num_new_renders, 1, 1, 3], cfg.background_min, cfg.background_max)

    new_std_images = set_background(new_std_images, mask, color)
    new_adv_images = set_background(new_adv_images, mask, color)

    return new_std_images, new_adv_images


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


def apply_print_error(std_texture, adv_texture):
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
    std_texture = transform(std_texture, multiplier, addend)
    adv_texture = transform(adv_texture, multiplier, addend)

    return std_texture, adv_texture


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
    std_images = transform(std_images, multiplier, addend)
    adv_images = transform(adv_images, multiplier, addend)

    gaussian_noise = tf.random.truncated_normal(
        shape=tf.shape(std_images),
        stddev=tf.random.uniform([1], maxval=cfg.stddev)
    )

    std_images += gaussian_noise
    adv_images += gaussian_noise

    return std_images, adv_images


def transform(x, a, b):
    """Apply transform a * x + b element-wise.

     Parameters
    ----------
        x : tensor
        a : tensor
        b : tensor
    """
    return tf.add(tf.multiply(a, x), b)


def normalize(std_images, adv_images):
    std_images_minimums = tf.reduce_min(std_images, axis=[1, 2, 3], keepdims=True)
    adv_images_minimums = tf.reduce_min(adv_images, axis=[1, 2, 3], keepdims=True)

    std_images_maximums = tf.reduce_max(std_images, axis=[1, 2, 3], keepdims=True)
    adv_images_maximums = tf.reduce_max(adv_images, axis=[1, 2, 3], keepdims=True)

    minimum = tf.minimum(std_images_minimums, adv_images_minimums)
    maximum = tf.maximum(std_images_maximums, adv_images_maximums)

    minimum = tf.minimum(minimum, 0)
    maximum = tf.maximum(maximum, 1)

    return (std_images - minimum) / (maximum - minimum), (adv_images - minimum) / (maximum - minimum)
