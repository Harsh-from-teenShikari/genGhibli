import abc
from imgprocalgs.algorithms.utilities import Image

class BaseAlgorithm(metaclass=abc.ABCMeta):
    """ Base class for all algorithms """
    def __init__(self, image_path: str, destination_path: str):
        self.image_path = image_path
        self.destination_path = destination_path
        self.image = Image(image_path)

    @abc.abstractmethod
    def process(self):
        """ Run the algorithm processing """
        pass
