.. image:: https://raw.githubusercontent.com/grimmlab/ForeTiS/master/docs/image/Logo_ForeTiS_Text.png
    :width: 600
    :alt: ForeTiS

ForeTiS: A Forecasting Time Series Framework
===================================================
ForeTiS is a Python framework that enables the rigorous training, comparison and analysis of predictions for a variety of different models.
It is designed for seasonal time-series data.
ForeTiS includes multiple state-of-the-art prediction models or machine learning methods, respectively.
These range from classical models, such as regularized linear regression over ensemble learners, e.g. XGBoost,
to deep learning-based architectures, such as Multilayer Perceptron (MLP).
To enable automatic hyperparameter optimization, we leverage state-of-the-art and efficient Bayesian optimization techniques.
Besides the named features, the forecasting models can adapt to changing trends and patterns in the data by being
regularly updated in different periods with new data by so-called periodical refitting. Doing so can simulate a
potential scenario for a productive operation.
The subsequent scheme gives an overview of the structure of ForeTiS: In preparation, we summarize the fully automated
and configurable data preprocessing and feature engineering. We already integrated several time series forecasting models
in model from which the user can choose. Furthermore, the design of this module enables a straightforward
integration of new prediction models. For automated hyperparameter optimization, we leverage state-of-the-art
Bayesian optimization using the Python package Optuna. With the module testing, we allow the user to test
different refitting procedures. Finally, we provide several methods to analyze results in evaluation. To start the
optimization pipeline, users only need to supply a CSV file containing the data and a configuration file that enables pipeline customization. This design allows end users to apply time series forecasting with only a single-line
command. In addition, we support researchers aiming to develop new forecasting methods with quick integration
in a reliable framework and benchmarking against existing approaches.

.. image:: https://raw.githubusercontent.com/grimmlab/ForeTiS/master/docs/image/ForeTiS_Figure1_raw.png
    :width: 600
    :alt: scheme of ForeTiS
    :align: center

In addition, our framework is designed to allow an easy and straightforward integration of further prediction models.

For more information, installation guides, tutorials and much more, see our documentation: https://ForeTiS.readthedocs.io/

Feel free to use the Docker workflow as described in our documentation: https://ForeTiS.readthedocs.io/en/latest/tutorials.html#howto-run-ForeTiS-using-docker

Contributors
----------------------------------------

This pipeline is developed and maintained by members of the `Bioinformatics lab <https://bit.cs.tum.de>`_ lead by `Prof. Dr. Dominik Grimm <https://bit.cs.tum.de/team/dominik-grimm/>`_:

- `Florian Haselbeck, M.Sc. <https://bit.cs.tum.de/team/florian-haselbeck/>`_
- `Josef Eiglsperger, M.Sc. <https://bit.cs.tum.de/team/josef-eiglsperger/>`_

Citation
---------------------

When using ForeTiS, please cite our publication:

**ForeTiS: A comprehensive time series forecasting framework in Python.**
Josef Eiglsperger*, Florian Haselbeck* and Dominik G. Grimm
*Machine Learning with Applications, 2023.* [doi: 10.1016/j.mlwa.2023.100467](https://doi.org/10.1016/j.mlwa.2023.100467)
**These authors have contributed equally to this work and share first authorship.*

.. toctree::
   :titlesonly:

   install
   quickstart
   tutorials
   data
   models
   API Documentation <autoapi/ForeTiS/index>
