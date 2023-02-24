import joblib
import tensorflow as tf

from . import _base_model, _tensorflow_model



def load_model(path: str, filename: str) -> _base_model.BaseModel:
    """
    Load persisted model
    :param path: path where the model is saved
    :param filename: filename of the model
    :return: model instance
    """
    model = joblib.load(path.joinpath(filename))
    # special case for loading tensorflow optimizer
    # if issubclass(type(model), _tensorflow_model.TensorflowModel):
    #     model.optimizer = tf.keras.optimizers.deserialize(model.optimizer)
    return model
