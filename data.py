import os
import csv

import numpy as np
from PIL import Image

class Model3D:
    def __init__(self, folder, data_dir, index):
        self.name = folder
        self.index = index
        absolute_model_path = os.path.join(data_dir, self.name)

        self.raw_texture = Model3D._get_texture(Model3D._get_texture_path(absolute_model_path))
        self.obj_path = os.path.join(absolute_model_path, "{}.obj".format(self.name))
        self.labels = Model3D._load_labels(absolute_model_path)

    def __str__(self):
        return "{}: labels {}".format(self.name, self.labels)

    @staticmethod
    def _get_texture(image_path):
        """Read texture from file and return it in the appropriate format.

        Parameters
        ----------
        path : String
            Absolute path to dataset sample folder.

        Returns
        -------
        Numpy array
            Numpy array representing the raw texture. Has shape width x height x 3.
        """
        texture_image = Image.open(image_path)

        # convert image to a numpy array with float values
        raw_texture = np.array(texture_image).astype(np.float32)
        texture_image.close()
        # some raw textures have an alfa channel too, we only want three colour channels
        raw_texture = raw_texture[:, :, :3]
        # normalise pixel vaues to between 0 and 1
        raw_texture = raw_texture / 255.0

        return raw_texture

    @staticmethod
    def _get_texture_path(path):
        """Determines if texture is a jpg or png file, and returns absolute path to texture file.

        Parameters
        ----------
        path : String
            Absolute path to dataset sample folder.

        Returns
        -------
        String
            Absolute path to texture file.
        """
        if not os.path.isdir(path):
            raise ValueError("The given absolute path is not a directory!")

        for file in os.listdir(path):
            if file.endswith(".jpg"):
                return os.path.join(path, file)
            elif file.endswith(".png"):
                return os.path.join(path, file)

        raise ValueError("No jpg or png files found in the given directory!")

    @staticmethod
    def _load_labels(path):
        """Reads labels of a certain sample from the dataset and returns them.

        Parameters
        ----------
        path : String
            Absolute path to dataset sample folder.

        Returns
        -------
        String or list
            Returns a list of integers, or if this is the dog model, just returns "dog" as a label.
        """
        if not os.path.isdir(path):
            raise ValueError("The given absolute path is not a directory!")

        labels_file_path = os.path.join(path, "labels.txt")
        try:
            labels_file = open(labels_file_path)
        except FileNotFoundError:
            raise FileNotFoundError("No txt files found in the given path! Can not find labels!")

        # labels are written only on the first line of the file, we only read the first line
        labels = next(csv.reader(labels_file, delimiter=','))
        # German shepherd model has all 120+ dog labels as true labels, that is encoded only as "dog" to save
        # make things easier
        if labels[0] == 'dog':
            return labels[0]
        else:
            try:
                int_labels = [int(label) for label in labels]
                return int_labels
            except ValueError as e:
                print("Original exception message: {}".format(str(e)))
                raise ValueError("A label of {} does not represent an int!".format(path))
            finally:
                labels_file.close()


def get_object_folders(data_dir):
    """Returns a list of all folders in the given folder.

    Parameters
    ----------
    path : String
        Absolute path to dataset sample folder.

    Returns
    -------
    List of strings
        Returns a list of with the name of each folder.
    """
    if not os.path.isdir(data_dir):
        raise ValueError("The given data path is not a directory!")

    return [folder for folder in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, folder))]


def load_dataset(data_dir):
    object_folders = get_object_folders(data_dir)
    models = [Model3D(folder, data_dir, i) for i, folder in enumerate(object_folders)]
    for model in models:
        print(str(model))

    return models



