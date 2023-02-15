.. image:: https://raw.githubusercontent.com/grimmlab/ForeTiS/main/docs/image/Logo_ForeTiS_Text.png
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
The subsequent scheme gives an overview on the hyperparameter optimization, training, and testing processes in ForeTiS.

.. image:: https://raw.githubusercontent.com/grimmlab/ForeTiS/master/docs/image/Algo.png
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

A manuscript for publishing ForeTiS as a further scientific paper is currently under preparation.

.. toctree::
   :titlesonly:

   install
   quickstart
   tutorials
   data
   models
   API Documentation <autoapi/ForeTiS/index>
