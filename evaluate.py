import os

import numpy as np
import tensorflow as tf
from PIL import Image

import data
import diff_rendering
import renderer
from config import cfg
from statistics import mean

decode_predictions = tf.keras.applications.imagenet_utils.decode_predictions

def render_image_for_texture(std_texture, adv_texture, renderer):
    width = std_texture.shape[1]
    height = std_texture.shape[0]

    uv_map = renderer.render(1)
    uv_map = uv_map * np.asarray([width - 1, height - 1], dtype=np.float32)

    std_image, adv_image = diff_rendering.render(std_texture, adv_texture, uv_map)

    # convert tensors to numpy arrays and discard the batch dimension
    std_image = std_image.numpy()[0]
    adv_image = adv_image.numpy()[0]

    return std_image, adv_image

def save_rendered_images(std_image, adv_image, model_name, target_label, num_image):
    if not os.path.exists('./evaluation_images/normal/{}'.format(model_name)):
        os.makedirs('./evaluation_images/normal/{}'.format(model_name))

    if not os.path.exists('./evaluation_images/normal/{}/{}'.format(model_name, target_label)):
        os.makedirs('./evaluation_images/normal/{}/{}'.format(model_name, target_label))

    if not os.path.exists('./evaluation_images/adv/{}'.format(model_name)):
        os.makedirs('./evaluation_images/adv/{}'.format(model_name))

    if not os.path.exists('./evaluation_images/adv/{}/{}'.format(model_name, target_label)):
        os.makedirs('./evaluation_images/adv/{}/{}'.format(model_name, target_label))

    # Pillow only accepts numpy arrays with integer values as valid images
    std_image = (std_image * 255).astype('uint8')
    adv_image = (adv_image * 255).astype('uint8')

    Image.fromarray(std_image, 'RGB').save('./evaluation_images/normal/{}/{}/image_{}.jpg'.format(
        model_name, target_label, num_image))
    Image.fromarray(adv_image, 'RGB').save('./evaluation_images/adv/{}/{}/image_{}.jpg'.format(
        model_name, target_label, num_image))

def get_tfr_and_accuracy(model, target_label, predictions):
    label_predictions = [np.argmax(prediction) for prediction in predictions]
    predictions = decode_predictions(predictions)

    accuracy = mean([is_prediction_true(model.labels, predicted_label) for predicted_label in label_predictions])
    tfr = sum([target_label == predicted_label for predicted_label in label_predictions])
    tfr  = tfr / len(label_predictions)

    return accuracy, tfr

def is_prediction_true(true_labels, predicted_label):
    if true_labels == "dog":
        # dog model has all 120 dog breed and dog-like animals as true labels
        if 150 < predicted_label < 276:
            return True
    # even if object only has one true label, it is still represented as a list with just one element
    elif type(true_labels) == list:
        if predicted_label in true_labels:
            return True
    else:
        raise ValueError("true labels list for a sample should be either \"dog\" or a list of ints.")

    # if it has not returned so far, then the prediction is incorrect
    return False

def parse_adv_texture_file_name(file_name):
    # removed extension from image file name
    file_name, _ = os.path.splitext(file_name)
    file_name_split = file_name.split('_')

    target_label = int(file_name_split[-3])
    num_steps = int(file_name_split[-1])

    index_first_digit = get_index_first_digit(file_name)
    # before the first digit in the file name, there is an underscore, which is not part of the name. Therefore we
    # slice until index_first_digit - 1
    model_name = file_name[:(index_first_digit - 1)]

    return model_name, target_label, num_steps


def get_index_first_digit(string):
    for i, character in enumerate(string):
        if character.isdigit():
            return i
    raise ValueError("The given string is expectde to have numbers in it!")


if __name__ == '__main__':
    models = data.load_dataset("./dataset")

    uv_renderer = renderer.Renderer((299, 299))
    uv_renderer.set_parameters(
        camera_distance=(cfg.camera_distance_min, cfg.camera_distance_max),
        x_translation=(cfg.x_translation_min, cfg.x_translation_max),
        y_translation=(cfg.y_translation_min, cfg.y_translation_max)
    )

    victim_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights='imagenet',
        classes=1000,
        classifier_activation='softmax'
    )
    victim_model.compile(optimizer='adam',
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                         metrics=['accuracy'])
    normalization_layer = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1)

    for image_file_name in os.listdir("./adv_textures"):
        adv_texture = data.Model3D._get_texture("./adv_textures/{}".format(image_file_name))

        # extract information from file name of adversarial texture, inclduing which model and target label is the
        # texture for
        current_model_name, current_target_label, num_steps = parse_adv_texture_file_name(image_file_name)

        # find the model that the adversarial texture was made for
        current_model = next(x for x in models if x.name == current_model_name)
        # get the normal texture of the model
        std_texture = current_model.raw_texture
        # load the appropriate model into the renderer
        uv_renderer.load_obj(current_model.obj_path)

        print("Creating evaluation renders for model {}, target label {}".format(current_model_name,
                                                                                 current_target_label))
        std_images = []
        adv_images = []
        for i in range(100):
            std_image, adv_image = render_image_for_texture(std_texture, adv_texture, uv_renderer)
            std_images.append(std_image)
            adv_images.append(adv_image)

            save_rendered_images(std_image, adv_image, current_model_name, current_target_label, i)

        # convert list of numpy images to one single numpy array
        std_images = np.stack(std_images, axis=0)
        adv_images = np.stack(adv_images, axis=0)
        # scale images from 0 to 1 values to values between -1 and 1
        std_images = 2 * std_images - 1
        adv_images = 2 * adv_images - 1

        predictions = victim_model.predict(std_images, batch_size=1)
        tfr, accuracy = get_tfr_and_accuracy(current_model, current_target_label, predictions)
        predictions = victim_model.predict(adv_images, batch_size=1)
        tfr, accuracy = get_tfr_and_accuracy(current_model, current_target_label, predictions)

        print("test")