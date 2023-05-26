Baseline Models
=============================================
Subsequently, we give details on the baseline models that are integrated in ForeTiS.

Often, the Root-Mean-Squared-Error (RMSE) gets used as an evaluation metric in forecasting tasks. But due
to the quadratic term, RMSE is sensitive to outliers. On the basis of these weaknesses and the lack of a universal
evaluation metric for forecasting, it is common to assess performance compared to baseline methods.
In the following, the in ForeTiS integrated four baseline models are listed:

    .. math::
            Average Historical: \hat{y}_t=\frac{1}{t-1} \sum_{t=1}^{t-1} y_t

    .. math::
            Average Moving/Random Walk: \hat{y}_t=\frac{1}{w} \sum_{t=t-w}^{t-1} y_t

    .. math::
            Average Seasonal: \hat{y}_t=\frac{1}{m} \sum_{t=t-m}^{t-1} y_t

    .. math::
            Average Seasonal Lag: \hat{y}_t=\frac{1}{w} \sum_{t=t-m-w}^{t-m-1} y_t

All these four approaches - averagehistorical, averagemoving, averageseasonal, and averageseasonallag - are currently implemented in ForeTiS.

The following code block shows the implementation of averagemoving in `averagemoving.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/averagemoving.py>`_.

    .. code-block::

        class AverageMoving(_baseline_model.BaselineModel):
            """
            Implementation of a class for AverageMoving.

            See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
            """

            def define_model(self):
                """
                Definition of the actual prediction model.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
                """
                self.window = self.suggest_hyperparam_to_optuna('window')
                return AverageMoving

            def define_hyperparams_to_tune(self) -> dict:
                """
                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
                """
                return {
                    'window': {
                        'datatype': 'int',
                        'lower_bound': 1,
                        'upper_bound': 20
                    }
                }

The other baseline models are implemented in a separate files containing very similar code.
Its implementation can be found in `averagehistorical.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/averagehistorical.py>`_,
`averageseasonal.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/averageseasonal.py>`_, and
`averageseasonallag.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/averageseasonallag.py>`_.


**References**

1. Hyndman, R.J., Koehler, A.B., 2006. Another look at measures of forecast accuracy. International Journal of Forecasting 22, 679–688.