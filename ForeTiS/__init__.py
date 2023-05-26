import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import ForeTiS.evaluation as evaluation
    import ForeTiS.model as model
    import ForeTiS.utils as utils
    import ForeTiS.optimization as optimization
    import ForeTiS.preprocess as preprocess

    from . import optim_pipeline

__version__ = "0.0.1"
__author__ = 'Josef Eiglsperger, Florian Haselbeck, Dominik G. Grimm'
__credits__ = 'GrimmLab @ TUM Campus Straubing (https://bit.cs.tum.de/)'