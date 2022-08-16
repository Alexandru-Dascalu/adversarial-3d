from pyrr import Matrix44
import moderngl
from objloader import Obj
import os
import numpy as np
import tensorflow as tf
from PIL import Image

import renderer


class TextureRenderer:
    window_size = (1199, 1199)
    aspect_ratio = 1

    resource_dir = os.path.normpath(os.path.join(__file__, '../'))
    texture_path = 'image_dir/adv_1980.jpg'
    obj_path = '3d_model/barrel.obj'
    output_path = "adv"

    def __init__(self):
        self.ctx = moderngl.create_standalone_context(require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.fbo = self.ctx.framebuffer(
            [self.ctx.renderbuffer(TextureRenderer.window_size)],
            self.ctx.depth_renderbuffer(TextureRenderer.window_size)
        )

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_vert;
                in vec2 in_text;
                in vec3 in_norm;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text_coord;

                void main() {
                    v_vert = in_vert;
                    v_norm = in_norm;
                    v_text_coord = in_text;
                    gl_Position = Mvp * vec4(v_vert, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                uniform sampler2D Texture;
                uniform vec4 Color;

                in vec3 v_vert;
                in vec2 v_text_coord;

                out vec4 f_color;

                void main() {
                    vec3 color = texture(Texture, v_text_coord).rgb;
                    color = color * (1.0 - Color.a) + Color.rgb * Color.a;
                    f_color = vec4(color, 1.0);
                }
            ''',
        )

        self.texture = self.load_texture(TextureRenderer.texture_path)
        self.vao = []
        self.load_obj(TextureRenderer.obj_path)

        self.color = self.prog['Color']
        self.color.value = (1.0, 1.0, 1.0, 0.0)
        self.mvp = self.prog['Mvp']

    def load_obj(self, file_path):
        """
        Load 3D model from .obj file and create vertex array based on it.

        Parameters
        ----------
        file_path : string
            Path to .obj file of the object that will be rendered.
        """
        if not os.path.isfile(file_path):
            print('{} is not an existing regular file!'.format(file_path))
            return

        obj = Obj.open(file_path)

        vbo = self.ctx.buffer(obj.pack('vx vy vz tx ty'))
        # TODO: not very efficient, consider using an element index array later
        self.vao = self.ctx.simple_vertex_array(
            self.prog,
            vbo,
            "in_vert", "in_text"
        )

    def load_texture(self, path):
        texture_image = Image.open(path)
        texture_image = texture_image.transpose(Image.FLIP_TOP_BOTTOM)
        texture_size = texture_image.size

        raw_image = texture_image.tobytes()
        texture_image.close()

        return self.ctx.texture(texture_size, 3, raw_image)

    def render(self):
        for i in range(100):
            self.ctx.clear(1.0, 1.0, 1.0)

            rotation = Matrix44.from_matrix33(
                renderer.Renderer.rand_rotation_matrix()
            )
            translation = Matrix44.from_translation((
                np.random.uniform(-0.05, 0.05),
                np.random.uniform(-0.05, 0.05),
                0
            ))
            proj = Matrix44.perspective_projection(45.0, self.aspect_ratio, 0.1, 1000.0)
            lookat = Matrix44.look_at(
                (0.1, 0, np.random.uniform(1.8, 2.3)),
                (0, 0, 0),
                (0.0, 1.0, 0.0),
            )

            self.fbo.use()
            self.fbo.clear(0.0, 0.0, 0.0, 1.0)

            self.mvp.write((proj * lookat * translation * rotation).astype('f4'))

            self.texture.use()
            self.vao.render()

            image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
            image.save('evaluation_images/{}/image_{}.jpg'.format(TextureRenderer.output_path, i))

    @classmethod
    def run(cls):
        loader = TextureRenderer()
        loader.render()


def evaluate(folder):
    data = tf.keras.preprocessing.image_dataset_from_directory(
        'evaluation_images/{}'.format(folder),
        labels=None,
        label_mode=None,
        image_size=(299, 299),
        batch_size=None)

    normalization_layer = tf.keras.layers.Rescaling(1. / 255)
    data = np.asarray([normalization_layer(image_batch) for image_batch in data])

    model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights='imagenet',
        classes=1000,
        classifier_activation='softmax'
    )
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    correct_test_labels = np.asarray([427] * 100)
    adv_test_labels = np.asarray([463] * 100)

    predictions = model.predict(data, batch_size=1)
    print([np.argmax(prediction) for prediction in predictions])

    _, accuracy = model.evaluate(data, correct_test_labels,  batch_size=1)
    print("Normal accuracy: {}".format(accuracy * 100))

    _, accuracy = model.evaluate(data, adv_test_labels, batch_size=1)
    print("Target label accuracy: {}".format(accuracy * 100))


def get_top_k_predictions(predictions, k=5):
    assert len(predictions) == 100

    count_dict = dict()
    unique_predictions = set()
    for prediction in predictions:
        unique_predictions.add(prediction)
        if prediction in count_dict:
            count_dict[prediction] += 1
        else:
            count_dict[prediction] = 1

    sorted_predictions = list(unique_predictions)
    sorted_predictions.sort(key=lambda x: -count_dict[x])
    return sorted_predictions[:k]


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    print(tf.test.is_gpu_available())
    print(tf.test.is_built_with_cuda())

    print("Adversarial texture:")
    TextureRenderer.texture_path = 'image_dir/adv_1980.jpg'
    TextureRenderer.output_path = '3d_model/barrel.obj'
    TextureRenderer.output_path = 'adv'
    TextureRenderer.run()
    evaluate(TextureRenderer.output_path)

    print("Normal texture:")
    TextureRenderer.texture_path = '3d_model/barrel.jpg'
    TextureRenderer.output_path = 'normal'
    TextureRenderer.run()
    evaluate(TextureRenderer.output_path)
