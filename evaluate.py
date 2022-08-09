from pyrr import Matrix44
import moderngl
from objloader import Obj
import os
import numpy as np
import tensorflow as tf
from PIL import Image

import renderer


class LoadingOBJ:
    gl_version = (3, 3)
    title = "Loading OBJ"
    window_size = (1199, 1199)
    aspect_ratio = 1
    resizable = True

    resource_dir = os.path.normpath(os.path.join(__file__, '../'))
    texture_path = 'image_dir/adv_1980.jpg'
    output_path = "adv"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ctx = moderngl.create_standalone_context(require=330)
        # why use depth test and face culling? Is it necessary to discard polygons which will not be visible in the
        # image, as we do not need those for computing the UV mapping?
        # self.ctx.enable(moderngl.DEPTH_TEST)
        # self.ctx.enable(moderngl.CULL_FACE)

        self.texture = self.load_texture(LoadingOBJ.texture_path)

        # Create frame buffer, where the scene will be rendered. Texture is empty, as we do not need the actual texture
        # of the model for calculating the UV mapping.
        self.fbo = self.ctx.simple_framebuffer((1199, 1199))

        self.prog = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                    gl_Position = Mvp * vec4(in_position, 1.0);
                }
            ''',
            fragment_shader='''
                #version 330

                uniform sampler2D Texture;
                uniform vec4 Color;

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                    vec3 color = texture(Texture, v_text).rgb;
                    color = color * (1.0 - Color.a) + Color.rgb * Color.a;
                    f_color = vec4(color, 1.0);
                }
            ''',
        )

        self.vao = []
        self.load_obj('3d_model/barrel.obj')
        self.color = self.prog['Color']
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

        # TODO: not very efficient, consider using an element index array later
        # make vertex array with two attributes, in_vert as vec3 and in_text as vec_2
        self.vao = self.ctx.simple_vertex_array(
            self.prog,
            self.ctx.buffer(obj.pack('vx vy vz tx ty')),
            "in_position", "in_texcoord_0"
        )

    def load_texture(self, path):
        image = Image.open(path)
        raw_image = image.tobytes()
        image.close()

        return self.ctx.texture((2048, 2048), 3, raw_image)

    def render(self):
        for i in range(100):
            self.ctx.clear(1.0, 1.0, 1.0)
            self.ctx.enable(moderngl.DEPTH_TEST)

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

            self.color.value = (1.0, 1.0, 1.0, 0.0)
            self.mvp.write((proj * lookat * translation * rotation).astype('f4'))

            self.texture.use()
            self.vao.render()

            image = Image.frombytes('RGB', self.fbo.size, self.fbo.read(), 'raw', 'RGB', 0, -1)
            image.save('evaluation_images/{}/image_{}.jpg'.format(LoadingOBJ.output_path, i))

    @classmethod
    def run(cls):
        loader = LoadingOBJ()
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


if __name__ == '__main__':
    print(tf.config.list_physical_devices('GPU'))
    print(tf.test.is_gpu_available())
    print(tf.test.is_built_with_cuda())

    print("Adversarial texture:")
    LoadingOBJ.texture_path = 'image_dir/adv_1980.jpg'
    LoadingOBJ.output_path = 'adv'
    LoadingOBJ.run()
    evaluate(LoadingOBJ.output_path)

    print("Normal texture:")
    LoadingOBJ.texture_path = '3d_model/barrel.jpg'
    LoadingOBJ.output_path = 'normal'
    LoadingOBJ.run()
    evaluate(LoadingOBJ.output_path)