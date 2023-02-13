Linear Regression
=============================================
Subsequently, we give details on the regularized linear regression approaches that are integrated in ForeTiS.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this text.
For our implementation, we use the machine learning framework scikit-learn, which also provides a `user guide for these models <https://scikit-learn.org/stable/modules/linear_model.html>`_.

With respect to regularized linear regressions models, the model weights can be optimized by minimizing
the deviation between predicted and true phenotypic values, often with considering an additive penalty term for regularization:

    .. math::
       \mathrm{argmin}_{\mathbf{w}} \frac{1}{2} |\mathbf{y} - \mathbf{X^{\ast}} \mathbf{w} |_2^2 + \alpha \Omega(\mathbf{w})

In case of the Least Absolute Shrinkage and Selection Operator, usually abbreviated with LASSO,
the L1-norm, so the sum of the absolute value of the weights, is used for regularization. This constraint
usually leads to sparse solutions forcing unimportant weights to zero. Intuitively speaking, this can be seen as an automatic feature selection.
The L2-norm, also known as the Euclidean norm, is defined as the square root of the summed up quadratic weights.
Regularized linear regression using the L2-norm is called Ridge Regression. This penalty term has the effect
of grouping correlated features. Elastic Net combines both the L1- and the L2-norm, introducing a further hyperparameter
controlling the influence of each of the two parts.

All these three approaches - LASSO, Ridge and Elastic Net Regression - are currently implemented in ForeTiS.

The following code block shows the implementation of LASSO in `lasso.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/lasso.py>`_.

    .. code-block::

        class Lasso(_sklearn_model.SklearnModel):
            """
            Implementation of a class for Lasso.

            See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
            """

            def define_model(self) -> sklearn.linear_model.Lasso:
                """
                Definition of the actual prediction model.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
                """
                # all hyperparameters defined for XGBoost are suggested for optimization
                self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
                if self.standardize_X:
                    self.x_scaler = sklearn.preprocessing.StandardScaler()

                alpha = self.suggest_hyperparam_to_optuna('alpha')
                params = {}
                params.update({'random_state': 42})
                params.update({'fit_intercept': True})
                params.update({'copy_X': True})
                params.update({'precompute': False})
                params.update({'max_iter': 10000})
                params.update({'tol': 1e-4})
                params.update({'warm_start': False})
                params.update({'positive': False})
                params.update({'selection': 'cyclic'})
                return sklearn.linear_model.Lasso(alpha=alpha, **params)

            def define_hyperparams_to_tune(self) -> dict:
                """
                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
                """
                return {
                    'alpha': {
                        'datatype': 'float',
                        'lower_bound': 10**-3,
                        'upper_bound': 10**3,
                        'log': True
                    },
                    'standardize_X': {
                        'datatype': 'categorical',
                        'list_of_values': [True, False]
                    }
                }

The other two regression models are implemented in a separate files containing very similar code.
Its implementation can be found in `elasticnet.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/elasticnet.py>`_.
Its implementation can be found in `ridge.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/ridge.py>`_.


**References**

1. Hastie, T., Tibshirani, R., & Friedman, J. H. (2009). The elements of statistical learning: data mining, inference, and prediction. 2nd ed. New York, Springer.
2. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. Journal of the Royal Statistical Society: Series B (Methodological), 58(1), 267–288.
3. Zou, H. and Hastie, T. (2005). Regularization and variable selection via the elastic net. Journal of the Royal Statistical Society, Series B, 67, 301–320.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
