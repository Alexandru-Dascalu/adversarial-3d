import numpy as np
import tensorflow as tf
# gpu = tf.config.list_physical_devices('GPU')[0]
# tf.config.set_logical_device_configuration(
#         gpu,
#         [tf.config.LogicalDeviceConfiguration(memory_limit=3900)])

from PIL import Image
from renderer import Renderer
from net import AdversarialNet
from config import cfg

LOGGING_ENABLED = False


def main():
    texture = Image.open(cfg.texture)
    height, width = texture.size

    renderer = Renderer(cfg.obj, (299, 299))
    renderer.set_parameters(
        camera_distance=(cfg.camera_distance_min, cfg.camera_distance_max),
        x_translation=(cfg.x_translation_min, cfg.x_translation_max),
        y_translation=(cfg.y_translation_min, cfg.y_translation_max)
    )

    texture = np.asarray(texture).astype(np.float32)[..., :3] / 255.0
    writer = tf.summary.create_file_writer(cfg.logdir)

    with tf.device("/device:GPU:0"):
        model = AdversarialNet(texture)
        for i in range(cfg.iterations):
            uv = renderer.render(cfg.batch_size) * \
                np.asarray([width - 1, height - 1], dtype=np.float32)

            model.optimisation_step(uv)

            if LOGGING_ENABLED:
                log_trainning(model, writer, i)
            print('Loss: {}'.format(model.loss))
            print('Diff: {}'.format(model.get_diff().sum()))
            print('Prediction:\n{}'.format(model.top_k_predictions))

            if i % 10 == 0:
                adv_texture = np.rint(model.adv_texture.numpy() * 255)
                adv_texture = Image.fromarray(adv_texture.astype(np.uint8))
                adv_texture.save('{}/adv_{}.jpg'.format(cfg.image_dir, i))

    writer.close()


def log_trainning(model, writer, epoch):
    with writer.as_default():
        tf.summary.image('train/std_images', model.std_images, step=epoch)
        tf.summary.image('train/adv_images', model.adv_images, step=epoch)
        tf.summary.scalar('train/loss', model.loss, step=epoch)
        tf.summary.histogram('train/top_k_predictions', model.top_k_predictions, step=epoch)
        writer.flush()


if __name__ == '__main__':
    main()
