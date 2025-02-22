import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

gpu = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_logical_device_configuration(
    gpu,
    [tf.config.LogicalDeviceConfiguration(memory_limit=3800)])


from renderer import Renderer
from net import AdversarialNet
import config
from config import cfg

FILE_LOGGING_ENABLED = False


def main():
    texture = Image.open(config.TEXTURE_PATH)
    height, width = texture.size

    assert height == width
    if height == 1024:
        config.BATCH_SIZE = 40
    elif height == 2048:
        config.BATCH_SIZE = 30
    else:
        raise ValueError("invalid texture size!")

    # create renderer used to make UV maps
    renderer = Renderer((299, 299))
    renderer.set_parameters(
        camera_distance=(cfg.camera_distance_min, cfg.camera_distance_max),
        x_translation=(cfg.x_translation_min, cfg.x_translation_max),
        y_translation=(cfg.y_translation_min, cfg.y_translation_max)
    )
    renderer.load_obj(config.OBJ_PATH)

    # convert image to numpy array and normalise it to values between 0 and 1
    texture = np.asarray(texture).astype(np.float32)[..., :3] / 255.0

    log_writer = None
    if FILE_LOGGING_ENABLED:
        log_writer = tf.summary.create_file_writer(cfg.logdir)

    with tf.device("/GPU:0"):
        # create the adversarial texture model that will be optimised. Holds all relevant tensors.
        model = AdversarialNet(texture)

        num_new_renders = int(np.ceil(config.BATCH_SIZE * (1 - cfg.batch_reuse_ratio)))
        print("New renders for each step: {}".format(num_new_renders))
        for i in range(cfg.iterations):
            # in the first step, there are no previous batch samples, and so we need to create a full batch.
            if i == 0:
                uv_mappings = renderer.render(config.BATCH_SIZE)
            else:
                uv_mappings = renderer.render(num_new_renders)

            # de-normalise UV mapping, so it has values from 0 to texture width-1 in uv[...,0] and 0 to height-1 in
            # uv[...,1], so 0 to 2047 by default.
            uv_mappings = uv_mappings * np.asarray([width - 1, height - 1], dtype=np.float32)

            # optimise adversarial texture
            model.optimisation_step(uv_mappings)

            if i % 200 == 0:
                log_training_to_console(model, i)
            if FILE_LOGGING_ENABLED:
                log_training_to_file(model, log_writer, i)

            # save intermediate adversarial textures, or the texture for the very last step
            if i % 200 == 0 or i == (cfg.iterations - 1):
                save_adv_texture(model, i)

            # check if the average loss in the last 400 steps is low enough to stop optimisation early
            if average_loss_under_threshold(model):
                # make sure to save the current adversarial texture before early stopping
                save_adv_texture(model, i)
                break

        plot_training_history(model)

    if log_writer is not None:
        log_writer.close()


def average_loss_under_threshold(model):
    """
    Calculate if the average loss of the past 400 steps if under the threshold.

    Parameters
    ----------
    model : AdversarialNet
        The adversarial model.

    Returns
    -------
    bool
        True if the average loss is under the threshold, False if not.
    """
    if len(model.main_loss_history) < 400:
        return False

    num_last_steps = 400
    loss_sum = sum(model.main_loss_history[-num_last_steps:])
    average_loss = loss_sum / num_last_steps

    if average_loss < cfg.loss_early_stopping_threshold:
        print("Early Stopping, average loss over past {} steps is {}".format(num_last_steps, average_loss))
        return True
    else:
        return False


def save_adv_texture(model, optimisation_step):
    adv_texture = np.rint(model.adv_texture.numpy() * 255)
    adv_texture = Image.fromarray(adv_texture.astype(np.uint8))
    adv_texture.save('{}/{}_{}_adv_{}.jpg'.format(cfg.image_dir, config.NAME, config.TARGET_LABEL, optimisation_step))


def log_training_to_console(model, step):
    print("Step: {}".format(step))
    print('Loss: {}'.format(model.main_loss_history[step]))
    print('Diff: {}'.format(model.get_diff().sum()))


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
        tf.summary.scalar('train/main_loss', model.main_loss_history, step=step)
        tf.summary.histogram('train/top_k_predictions', model.top_k_predictions, step=step)
        writer.flush()


def plot_training_history(adv_model):
    total_loss = [(main_loss + l2) for main_loss, l2 in zip(adv_model.main_loss_history, adv_model.l2_loss_history)]

    plt.plot(adv_model.main_loss_history, label="Main Loss")
    plt.plot(adv_model.l2_loss_history, label="L2 Loss")
    plt.plot(total_loss, label="Total Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss during training, {}, target {}".format(config.NAME, config.TARGET_LABEL))
    plt.legend()
    plt.show()

    plt.plot(adv_model.tfr_history, label="TFR")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("TFR during training, {}, target {}".format(config.NAME, config.TARGET_LABEL))
    plt.show()


if __name__ == '__main__':
    main()
