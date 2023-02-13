Bayesian Regression
=============================================
Subsequently, we give details on the bayesian regression approaches that are integrated in ForeTiS.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this text.
For our implementation, we use the machine learning framework scikit-learn, which also provides a `user guide for these models <https://scikit-learn.org/stable/modules/linear_model.html>`_.

Bayesian regression techniques can be understood as regularized linear regressions models (:ref:`Linear Regression`)
where the regularization parameter is not set by introducing uninformative priors over the hyper parameters of the model.
The L2-regularization used in Ridge regression is equivalent to finding a maximum a posteriori estimation under a
Gaussian prior over the coefficients w with precision

    .. math::
        \lambda^{-1}

Instead of setting lambda manually, it is possible to treat it as a random variable to be estimated from the data.

To obtain a fully probabilistic model, the output y is assumed to be Gaussian distributed around Xw:

    .. math::
        p(y|X,w,\alpha) = N(y|Xw,\alpha)

where alpha is again treated as a random variable that is to be estimated from the data.

The difference between ARD and Bayesian Ridge Regression is a different prior over w.

Both approaches - ARD and Bayesian Regression - are currently implemented in ForeTiS.

The following code block shows the implementation of ARD in `ard.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/ard.py>`_.

    .. code-block::

            def define_model(self) -> sklearn.linear_model.ARDRegression:
                """
                Definition of the actual prediction model.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
                """
                self.conf = True

                self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
                if self.standardize_X:
                    self.x_scaler = sklearn.preprocessing.StandardScaler()

                alpha_1 = self.suggest_hyperparam_to_optuna('alpha_1')
                alpha_2 = self.suggest_hyperparam_to_optuna('alpha_2')
                lambda_1 = self.suggest_hyperparam_to_optuna('lambda_1')
                lambda_2 = self.suggest_hyperparam_to_optuna('lambda_2')
                threshold_lambda = self.suggest_hyperparam_to_optuna('threshold_lambda')
                params = {}
                params.update({'fit_intercept': True})
                params.update({'n_iter': 10000})
                params.update({'tol': 1e-3})
                params.update({'copy_X': True})
                params.update({'verbose': False})
                params.update({'compute_score': False})
                return sklearn.linear_model.ARDRegression(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1,
                                                          lambda_2=lambda_2, threshold_lambda=threshold_lambda, **params)

            def define_hyperparams_to_tune(self) -> dict:
                """
                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
                """
                return {
                    'alpha_1': {
                        'datatype': 'float',
                        'lower_bound': 10**-3,
                        'upper_bound': 10**3,
                        'log': True
                    },
                    'alpha_2': {
                        'datatype': 'float',
                        'lower_bound': 10**-3,
                        'upper_bound': 10**3,
                        'log': True
                    },
                    'lambda_1': {
                        'datatype': 'categorical',
                        'list_of_values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
                    },
                    'lambda_2': {
                        'datatype': 'categorical',
                        'list_of_values': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
                    },
                    'threshold_lambda': {
                        'datatype': 'categorical',
                        'list_of_values': [1e2, 1e3, 1e4, 1e5, 1e6]
                    },
                    'standardize_X': {
                        'datatype': 'categorical',
                        'list_of_values': [True, False]
                    }
                }

The other regression model is implemented in a separate files containing very similar code.
Its implementation can be found in `bayesridge.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/bayesridge.py>`_.

**References**

1. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
2. D. J. C. MacKay, Bayesian Interpolation, Computation and Neural Systems, Vol. 4, No. 3, 1992.
3. M. E. Tipping, Sparse Bayesian Learning and the Relevance Vector Machine, Journal of Machine Learning Research, Vol. 1, 2001.
4. D. J. C. MacKay, Bayesian nonlinear modeling for the prediction competition, ASHRAE Transactions, 1994.
5. Christopher M. Bishop: Pattern Recognition and Machine Learning, 2006
6. Wipf, David, und Srikantan Nagarajan. „A New View of Automatic Relevance Determination“.
In Advances in Neural Information Processing Systems, Bd. 20. Curran Associates, Inc., 2007.
7. Tristan Fletcher: Relevance Vector Machines Explained