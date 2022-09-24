import os

import numpy as np
import tensorflow as tf
from PIL import Image

import data
import diff_rendering
import renderer
from config import cfg
from imagenet_labels import imagenet_labels


def render_images_for_texture(std_texture, adv_texture, uv_renderer, model_name, target_label):
    std_images = []
    adv_images = []
    for i in range(100):
        std_image, adv_image = render_image_for_texture(std_texture, adv_texture, uv_renderer)
        std_images.append(std_image)
        adv_images.append(adv_image)

        save_rendered_images(std_image, adv_image, model_name, target_label, i)

    # convert list of numpy images to one single numpy array
    std_images = np.stack(std_images, axis=0)
    adv_images = np.stack(adv_images, axis=0)
    # scale images from 0 to 1 values to values between -1 and 1, as that is what neural networks expect
    std_images = 2 * std_images - 1
    adv_images = 2 * adv_images - 1

    return std_images, adv_images


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

    accuracy = sum([is_prediction_true(model.labels, predicted_label) for predicted_label in label_predictions])
    accuracy = accuracy / len(label_predictions)

    tfr = sum([target_label == predicted_label for predicted_label in label_predictions])
    tfr  = tfr / len(label_predictions)

    return accuracy, tfr


def save_result(result_dict, result, model_name, target_label, tfr):
    """
    Save tfr or accuracy evaluation result to a dictionary based on the model and target label of the texture that was
    evaluated.
    """
    # initialise sub-dictionary for that particular model
    if model_name not in result_dict:
        result_dict[model_name] = dict()

    # initialise sub-dictionary for that particular target label in the sub-dictionary for the given model
    if target_label not in result_dict[model_name]:
        result_dict[model_name][target_label] = dict()

    # we may want to either save the TFR of adversarial texture or its accuracy
    if tfr:
        result_dict[model_name][target_label]['tfr'] = result
    else:
        result_dict[model_name][target_label]['accuracy'] = result


def save_num_steps(num_steps_dict, num_steps, model_name, target_label):
    """
    Save tfr or accuracy evaluation result to a dictionary based on the model and target label of the texture that was
    evaluated.
    """
    # initialise sub-dictionary for that particular model
    if model_name not in num_steps_dict:
        num_steps_dict[model_name] = dict()

    # initialise sub-dictionary for that particular target label in the sub-dictionary for the given model
    if target_label not in num_steps_dict[model_name]:
        num_steps_dict[model_name][target_label] = dict()

    num_steps_dict[model_name][target_label] = num_steps

def flatten_dict(result_dict, for_tfr):
    result_list = []
    for model in result_dict:
        for target_label in result_dict[model]:
            if for_tfr:
                result_list.append(result_dict[model][target_label]['tfr'])
            else:
                result_list.append(result_dict[model][target_label]['accuracy'])

    return result_list

def get_average_metric(results_dict, for_tfr):
    metric_sum = 0
    metric_count = 0

    for model_name in results_dict:
        metric_sum += get_average_metric_for_model(results_dict, model_name, for_tfr)
        metric_count += 1

    average = metric_sum / metric_count
    return average


def get_average_metric_for_model(results_dict, model_name, for_tfr):
    metric_sum = 0
    metric_count = 0

    for target_label in results_dict[model_name]:
        if for_tfr:
            metric_sum += results_dict[model_name][target_label]['tfr']
        else:
            metric_sum += results_dict[model_name][target_label]['accuracy']
        metric_count += 1

    average = metric_sum / metric_count
    return average


def get_average_num_steps(num_steps_dict, model_name):
    num_steps_sum = 0
    num_steps_count = 0

    for target_label in num_steps_dict[model_name]:
        num_steps_sum += num_steps_dict[model_name][target_label]
        num_steps_count += 1

    average = num_steps_sum / num_steps_count
    return average


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
    """
    Extracts useful information from the name of the files where the adversarial examples were saved.
    """
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
    """
    Returns index of the first digit to be found in a string.
    """
    for i, character in enumerate(string):
        if character.isdigit():
            return i
    raise ValueError("The given string is expectde to have numbers in it!")

def main():
    models = data.load_dataset("./dataset")

    # make renderer used for creating UV maps
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

    # dictionaries used to record results of the evaluation. Each dict has sub-dictionaries for each model and
    # target label
    normal_results = dict()
    adv_results = dict()
    num_steps_dict = dict()

    for image_file_name in os.listdir("./adv_textures"):
        adv_texture = data.Model3D._get_texture("./adv_textures/{}".format(image_file_name))

        # extract information from file name of adversarial texture, inclduing which model and target label is the
        # texture for
        current_model_name, current_target_label, num_steps = parse_adv_texture_file_name(image_file_name)
        save_num_steps(num_steps_dict, num_steps, current_model_name, current_target_label)

        # find the model that the adversarial texture was made for
        current_model = next(x for x in models if x.name == current_model_name)
        # get the normal texture of the model
        std_texture = current_model.raw_texture
        # load the appropriate model into the renderer
        uv_renderer.load_obj(current_model.obj_path)

        print("Creating evaluation renders for model {}, target label {} ({})".format(
            current_model_name, current_target_label, imagenet_labels[current_target_label]))
        std_images, adv_images = render_images_for_texture(std_texture, adv_texture, uv_renderer, current_model_name,
                                                           current_target_label)

        # evaluate renders with the normal image
        predictions = victim_model.predict(std_images, batch_size=1)
        accuracy, tfr = get_tfr_and_accuracy(current_model, current_target_label, predictions)
        # record results in dictionary and on the command line
        save_result(normal_results, tfr, current_model_name, current_target_label, tfr=True)
        save_result(normal_results, accuracy, current_model_name, current_target_label, tfr=False)
        print("Evaluating normal images: TFR: {}, Accuracy: {}".format(tfr, accuracy))

        # evaluate renders with the adversarial image
        predictions = victim_model.predict(adv_images, batch_size=1)
        accuracy, tfr = get_tfr_and_accuracy(current_model, current_target_label, predictions)
        # record results in dictionary and on the command line
        save_result(adv_results, tfr, current_model_name, current_target_label, tfr=True)
        save_result(adv_results, accuracy, current_model_name, current_target_label, tfr=False)
        print("Evaluating adversarial images: TFR: {}, Accuracy: {}\n".format(tfr, accuracy))

    print("Average accuracy for normal images: {}".format(get_average_metric(normal_results, for_tfr=False)))
    print("Average TFR for normal images: {}".format(get_average_metric(normal_results, for_tfr=True)))

    print("Average accuracy for adversarial images: {}".format(get_average_metric(adv_results, for_tfr=False)))
    print("Average TFR for adversarial images: {}\n".format(get_average_metric(adv_results, for_tfr=True)))

    for model_name in adv_results:
        print("Average TFR for model {}: {}".format(model_name, get_average_metric_for_model(adv_results, model_name,
                                                                                             for_tfr=True)))
        print("Average iterations for creating adversarial examples for model {}: {}".format(
            model_name, get_average_num_steps(num_steps_dict, model_name)))

if __name__ == '__main__':
    main()