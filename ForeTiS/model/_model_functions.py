import joblib

from . import _base_model


def load_model(path: str, filename: str) -> _base_model.BaseModel:
    """
    Load persisted model
    :param path: path where the model is saved
    :param filename: filename of the model
    :return: model instance
    """
    path = path + '/' if path[-1] != '/' else path
    model = joblib.load(path + filename)
    return model
