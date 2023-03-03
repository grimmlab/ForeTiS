Gaussian Process Regression
=============================================
Subsequently, we give details on the Gaussian Process Regression approaches that are integrated in ForeTiS.
References for a more detailed theoretical background can be found at the end of this page, which were also used for writing this text.
For our implementation, we use the machine learning framework GPflow, which also provides a `user guide for these models <https://www.gpflow.org/>`_.

Gaussian Processes (GP) are a generic supervised learning method. When designed to solve regression, it is called
Gaussian Process Regression (GPR). The prediction is probabilistic (Gaussian). Therefore, a empirical
confidence intervals can be computed.
EVent-triggered Augmented Refitting of Gaussian Process Regression for Seasonal Data (EVARS-GPR) andles sudden shifts
in the target variable scale of seasonal data by combining online change point detection with a refitting of the GPR
model using data augmentation for samples prior to a change point.

Both approaches - GPR and EVARS-GPR - are currently implemented in ForeTiS.

The following code block shows the implementation of GPR in `gprtf.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/gprtf.py>`_.
We completely outsourced the method in the parent class:
`_tensorflow_model.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/_tensorflow_model.py>`_.
This is becasue both approaches share the almost the same method, except for the predict method that is redefined in the EVARS-GPR class:
`evars-gpr.py <https://github.com/grimmlab/ForeTiS/blob/main/ForeTiS/model/evars-gpr.py>`_.

    .. code-block::

        class Gpr(_tensorflow_model.TensorflowModel):
            """
            Implementation of a class for Gpr.

            See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
            """

**References**

1. Haselbeck, F., Grimm, D.G., 2021. EVARS-GPR: EVent-triggered Augmented Refitting of Gaussian Process Regression for SeasonalData.
2. Alexander G. de G. Matthews, Mark van der Wilk, Tom Nickson, Keisuke. Fujii, Alexis Boukouvalas, Pablo León-Villagrá, Zoubin Ghahramani, and James Hensman. GPflow: A Gaussian process library using TensorFlow. Journal of Machine Learning Research, 18(40):1–6, apr 2017.
3. Mark van der Wilk, Vincent Dutordoir, ST John, Artem Artemev, Vincent Adam, and James Hensman. A framework for interdomain and multioutput Gaussian processes. arXiv:2003.01115, 2020.