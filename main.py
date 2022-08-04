import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_logical_device_configuration(
    gpu,
    [tf.config.LogicalDeviceConfiguration(memory_limit=3800)])

from PIL import Image
from renderer import Renderer
from net import AdversarialNet
from config import cfg

FILE_LOGGING_ENABLED = False


def main():
    texture = Image.open(cfg.texture)
    height, width = texture.size

    renderer = Renderer(cfg.obj, (299, 299))
    renderer.set_parameters(
        camera_distance=(cfg.camera_distance_min, cfg.camera_distance_max),
        x_translation=(cfg.x_translation_min, cfg.x_translation_max),
        y_translation=(cfg.y_translation_min, cfg.y_translation_max)
    )

    # convert image to numpy array and normalise it to values between 0 and 1
    texture = np.asarray(texture).astype(np.float32)[..., :3] / 255.0
    log_writer = tf.summary.create_file_writer(cfg.logdir)

    with tf.device("/GPU:0"):
        # create the adversarial texture model that will be optimised. Holds all relevant tensors.
        model = AdversarialNet(texture)
        loss_history = []
        tfr_history = []
        for i in range(cfg.iterations):
            # UV mapping is a numpy array of shape batch_size x texture_width x texture_height x 2
            uv = renderer.render(cfg.batch_size)
            # de-normalise UV mapping so it has values from 0 to texture width-1 in uv[...,0] and 0 to height-1 in
            # uv[...,1], so 0 to 2047 by default.
            uv = uv * np.asarray([width - 1, height - 1], dtype=np.float32)

            # optimise adversarial texture
            model.optimisation_step(uv)
            loss_history.append(model.loss)
            tfr_history.append(get_tfr(model.top_k_predictions))

            log_training_to_console(model, i)
            if FILE_LOGGING_ENABLED:
                log_training_to_file(model, log_writer, i)

            # save intermediate adversarial textures
            if i % 10 == 0:
                adv_texture = np.rint(model.adv_texture.numpy() * 255)
                adv_texture = Image.fromarray(adv_texture.astype(np.uint8))
                adv_texture.save('{}/adv_{}.jpg'.format(cfg.image_dir, i))

        plot_training_history(loss_history, tfr_history)

    log_writer.close()


def log_training_to_console(model, step):
    print("Step: {}".format(step))
    print('Loss: {}'.format(model.loss))
    print('Diff: {}'.format(model.get_diff().sum()))
    print('Prediction:\n{}'.format(model.top_k_predictions))


def log_training_to_file(model, writer, step):
    """
    Logs details of training steps to TF summary file writer.

    Parameters
    ----------
    model : AdversarialNet
        The Adversarial texture model currently being trained.
    writer : tf.summary.SummaryWriter
        The writer used for logging.
    step : int
        The training step currently being logged.
    """
    with writer.as_default():
        tf.summary.image('train/std_images', model.std_images, step=step)
        tf.summary.image('train/adv_images', model.adv_images, step=step)
        tf.summary.scalar('train/loss', model.loss, step=step)
        tf.summary.histogram('train/top_k_predictions', model.top_k_predictions, step=step)
        writer.flush()


def get_tfr(top_k_predictions):
    assert top_k_predictions.shape == (cfg.batch_size, 5)

    target_reached = 0
    for top_prediction in top_k_predictions[:, 0]:
        if top_prediction == cfg.target:
            target_reached += 1

    tfr = target_reached / cfg.batch_size
    return tfr


def plot_training_history(loss_history, tfr_history):
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss during training")
    plt.show()

    plt.plot(tfr_history, label="TFR")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("TFR during training")
    plt.show()


if __name__ == '__main__':
    main()
