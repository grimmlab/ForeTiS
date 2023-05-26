XGBoost
=============================================
Subsequently, we give details on our implementation of extreme gradient Boosting, usually abbreviated with XGBoost.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this text.
For our implementation, we use the library `xgboost <https://xgboost.readthedocs.io/en/stable/>`_,
which also provides a `user guide <https://xgboost.readthedocs.io/en/stable/python/index.html>`_.

XGBoost applies a technique called Boosting. Similar to Random Forest, XGBoost is also an ensemble learner, i.e. trying to
build a strong prediction model based on multiple weak learners.
But as a conceptual difference, weak learners in XGBoost are not independent. Instead, they are constructed
sequentially, with putting more focus on the errors of the current ensemble for the training of a new weak learner. With Gradient Boosting, the
sequential construction of the ensemble is formalized as a gradient descent algorithm on a loss function that needs to be minimized.

In comparison with Bagging, which is employed in Random Forest, Boosting aims to reduce bias instead of variance.
This might lead to overfitting, which is aimed to be prevented by certain measures. One example is constraining the weak learners,
so e.g. limiting the number of estimators or the depth of the Decision Trees. Further methods against overfitting
are similar to concepts of bagging, e.g. using random subsets of samples and features for the training of each weak learner.
Besides this, for XGBoost a learning rate shrinking the weights update for correcting ensemble errors during the learning process is typically used.

XGBoost is an efficient implementation that leverages Gradient Boosting, for which further details can be found in the
`original paper <https://dl.acm.org/doi/10.1145/2939672.2939785>`_. It has proven its predictive power in many application
areas, e.g. in Kaggle competitions.


For XGBoost, we use a specific library that is also available as a Python package. In the code block below,
you can see our implementation. Furthermore, we optimize several hyperparameters, such as the number of weak learners
(``n_estimators``) or the ``learning_rate``. A full explanation of all XGBoost parameters can be found in their
documentation: `XGBoost Parameter Guide <https://xgboost.readthedocs.io/en/stable/parameter.html>`_


    .. code-block::

        class XgBoost(_sklearn_model.SklearnModel):
            """
            Implementation of a class for XGBoost.

            See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
            """

            def define_model(self) -> xgboost.XGBModel:
                """
                Definition of the actual prediction model.

                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
                """
                # all hyperparameters defined for XGBoost are suggested for optimization
                params = self.suggest_all_hyperparams_to_optuna()
                params.update({'random_state': 42})
                params.update({'verbosity': 0})
                params.update({'objective': 'reg:squarederror'})
                params.update({'tree_method': 'auto'})
                return xgboost.XGBRegressor(**params)


            def define_hyperparams_to_tune(self) -> dict:
                """
                See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
                """
                return {
                    'n_estimators': {
                        'datatype': 'int',
                        'lower_bound': 500,
                        'upper_bound': 1000,
                        'step': 50
                    },
                    'max_depth': {
                        'datatype': 'int',
                        'lower_bound': 2,
                        'upper_bound': 10
                    },
                    'learning_rate': {
                        'datatype': 'float',
                        'lower_bound': 0.025,
                        'upper_bound': 0.3,
                        'step': 0.025
                    },
                    'gamma': {
                        'datatype': 'int',
                        'lower_bound': 0,
                        'upper_bound': 1000,
                        'step': 10
                    },
                    'subsample': {
                        'datatype': 'float',
                        'lower_bound': 0.05,
                        'upper_bound': 1.0,
                        'step': 0.05
                    },
                    'colsample_bytree': {
                        'datatype': 'float',
                        'lower_bound': 0.05,
                        'upper_bound': 1.0,
                        'step': 0.05
                    },
                    'reg_lambda': {
                        'datatype': 'float',
                        'lower_bound': 0,
                        'upper_bound': 1000,
                        'step': 1
                    },
                    'reg_alpha': {
                        'datatype': 'float',
                        'lower_bound': 0,
                        'upper_bound': 1000,
                        'step': 1
                    }
                }


**References**

1. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). New York, NY, USA: ACM.
