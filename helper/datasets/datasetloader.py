import numpy as np
import cv2
import os


class DatasetLoader:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    @classmethod
    def print_processing_details_to_console(cls, _index, _verbose, _image_paths):
        if _index > 0 and _index > 0 and (_index + 1) % _verbose == 0:
            print("[INFO] processed {}/{}".format(_index + 1,
                                                  len(_image_paths)))

    def validate_and_apply_preprocessor(self, image):
        if self.preprocessors is not None:
            return self.apply_preprocessor(image)
        return image

    def apply_preprocessor(self, image):
        for p in self.preprocessors:
            return p.preprocess(image)
        return image

    def preprocess_image_and_labels(self, image_paths, verbose):
        data = []
        labels = []

        for (index, _image_path) in enumerate(image_paths):
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            image = cv2.imread(_image_path)
            label = _image_path.split(os.path.sep)[-2]
            image = self.validate_and_apply_preprocessor(image)
            data.append(image)
            labels.append(label)
            self.print_processing_details_to_console(index, verbose, image_paths)
        return data, labels

    def load(self, image_paths, verbose=-1):
        data, labels = self.preprocess_image_and_labels(image_paths, verbose)
        return np.array(data), np.array(labels)
