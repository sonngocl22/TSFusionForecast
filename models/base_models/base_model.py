from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, config):
        """
        Initialize the base model

        :param config: configuration file which contains model parameters
        """
        self.config = config
        self.model = None
        self.retrain = False

    @abstractmethod
    def load_data(self, file_path):
        """
        Load data from a file. Must be done by a subclass.

        :param file_path: path to load the data
        :return:
        """
        pass

    @abstractmethod
    def train(self, train_data, validation_data=None):
        """
        Train model on training data. Must be done by a subclass.

        :param train_data: training data
        :param validation_data: validation data
        :return:
        """
        pass
    
    @abstractmethod
    def generate_meta(self, file_path):
        """
        Predict the future values of the time series from trained model and generate the meta data. Must be done by a subclass.

        :param file_path: path to save the meta data
        """
        pass

    @abstractmethod
    def save_model(self, file_path):
        """
        Save the model to a file. Must be done by a subclass.

        :param file_path: path to save the model
        :return:
        """
        pass

    @abstractmethod
    def load_model(self, file_path):
        """
        Load the model from a file. Must be done by a subclass.

        :param file_path: path to load the model
        :return:
        """
        pass