import joblib


from ..preprocess import base_dataset
from ..utils import helper_functions
from ..model import _base_model
def load_model(path: str, filename: str) -> _base_model.BaseModel:
    """
    Load persisted model
    :param path: path where the model is saved
    :param filename: filename of the model
    :return: model instance
    """
    model = joblib.load(path.joinpath(filename))
    return model

def retrain_model_with_results_file(featureset: base_dataset.Dataset, model) -> _base_model.BaseModel:
    """
    Retrain a model based on information saved in a results overview file.

    :param featureset: dataset for training
    :param model: model that you want to retrain on the whole data

    :return: retrained model
    """

    helper_functions.set_all_seeds()
    model.retrain(retrain=featureset)

    return model
