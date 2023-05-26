from . import _baseline_model


class AverageHistorical(_baseline_model.BaselineModel):
    """
    Implementation of a class for AverageHistorical.

    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

    def define_model(self):
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        return AverageHistorical

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        return {}
