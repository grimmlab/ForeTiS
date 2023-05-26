Prediction Models
===================
ForeTiS includes various time series forecasting models, both classical forecasting models as well as machine and deep learning-based methods.
In the following pages, we will give some details for all of the currently implemented models.
We further included a subpage explaining the Bayesian optimization that we use for our automatic hyperparameter search.

We provide both a workflow running ForeTiS with a command line interface using Docker and as a pip package, see the following tutorials for more details:

    - :ref:`HowTo: Run ForeTiS using Docker`

    - :ref:`HowTo: Use ForeTiS as a pip package`

In both cases, you need to select the prediction model you want to run - or also multiple ones within the same optimization run.
A specific prediction model can be selected by giving the name of the *.py* file in which it is implemented (without the *.py* suffix).
For instance, if you want to run Extreme Gradient Boost implemented in *xgboost.py*, you need to specify *xgboost*.

In the following table, we give the keys for all prediction models as well as links to detailed descriptions and the source code:

.. list-table:: Time Series Forecasting Models
   :widths: 25 15 20 20
   :header-rows: 1

   * - Model
     - Key in ForeTiS
     - Description
     - Source Code
   * - Automatic Relevance Determination Regression
     - ard
     - :ref:`Bayesian Regression`
     - `ard.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/ard.py>`_
   * - SARIMA
     - sarima
     - :ref:`SARIMA(X)`
     - `sarima.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/sarima.py>`_
   * - SARIMAX
     - sarimax
     - :ref:`SARIMA(X)`
     - `sarimax.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/sarimax.py>`_
   * - Average Historical
     - averagehsitorical
     - :ref:`Baseline Models`
     - `averagehistorical.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/averagehistorical.py>`_
   * - Average Moving
     - averagemoving
     - :ref:`Baseline Models`
     - `averagemoving.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/averagemoving.py>`_
   * - Average Seasonal
     - averageseasonal
     - :ref:`Baseline Models`
     - `averageseasonal.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/averageseasonal.py>`_
   * - Average Seasonal Lag
     - averageseasonallag
     - :ref:`Baseline Models`
     - `averageseasonallag.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/averageseasonallag.py>`_
   * - Bayesian Ridge Regression
     - bayesridge
     - :ref:`Bayesian Regression`
     - `bayesridge.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/bayesridge.py>`_
   * - Exponential Smoothing
     - es
     - :ref:`Exponential Smoothing`
     - `es.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/es.py>`_
   * - EVARS-GPR
     - evars-gpr
     - :ref:`Gaussian Process Regression`
     - `evars-gpr.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/evars-gpr.py>`_
   * - Gaussian Process Regression (TensorFlow Implemetation)
     - gprtf
     - :ref:`Gaussian Process Regression`
     - `gprtf.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/gprtf.py>`_
   * - Lasso Regression
     - lasso
     - :ref:`Linear Regression`
     - `lasso.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/lasso.py>`_
   * - Long Short-Term Memory (LSTM) Network
     - lstm
     - :ref:`LSTM Network`
     - `lstm.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/lstm.py>`_
   * - Bayesian Long Short-Term Memory (LSTM) Network
     - lstmbayes
     - :ref:`LSTM Network`
     - `lstmbayes.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/lstmbayes.py>`_
   * - Elastic Net Regression
     - elasticnet
     - :ref:`Linear Regression`
     - `elasticnet.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/elasticnet.py>`_
   * - Multilayer Perceptron
     - mlp
     - :ref:`Multilayer Perceptron`
     - `mlp.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/mlp.py>`_
   * - Bayesian Multilayer Perceptron
     - mlpbayes
     - :ref:`Multilayer Perceptron`
     - `bayesmlp.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/bayesmlp.py>`_
   * - Ridge Regression
     - ridge
     - :ref:`Linear Regression`
     - `ridge.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/ridge.py>`_
   * - XGBoost
     - xgboost
     - :ref:`XGBoost`
     - `xgboost.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/xgboost.py>`_

Also, ForeTiS contains the follwing additional models. These are easy to integrate by just copy and pasting them to the models folder.
They do not get support or updates anymore by the authors, as we decided to use other framework for the algorithms, but
they still should work.

.. list-table:: Additional Time Series Forecasting Models
   :widths: 25 15 20 20
   :header-rows: 1

   * - Model
     - Key in ForeTiS
     - Description
     - Source Code
   * - Gaussian Process Regression (with sklearn framework)
     - gpr_sklearn
     - :ref:`Gaussian Process Regression`
     - `gpr_sklearn.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/additionalmodels/gpr_sklearn.py>`_
   * - Bayesian Multilayer Perceptron (with IntelLabs BayesianTorch framework)
     - mlpbayes_intel
     - :ref:`Multilayer Perceptron`
     - `mlpbayes_intel.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/additionalmodels/mlpbayes_intel.py>`_
   * - Bayesian Long Short-Term Memory (LSTM) Network (with IntelLabs BayesianTorch framework)
     - lstmbayes_intel
     - :ref:`LSTM Network`
     - `lstmbayes_intel.py <https://github.com/grimmlab/ForeTiS/blob/master/ForeTiS/model/additionalmodels/lstmbayes_intel.py>`_

If you are interested in adjusting an existing model or its hyperparameters: :ref:`HowTo: Adjust existing prediction models and their hyperparameters`.

If you want to integrate your own prediction model: :ref:`HowTo: Integrate your own prediction model`.

.. toctree::
    :maxdepth: 4
    :hidden:

    models/baseline
    models/es
    models/sarima(x)
    models/linreg
    models/bayesreg
    models/gpr
    models/xgb
    models/lstm
    models/mlp
    models/hyperparam_optim