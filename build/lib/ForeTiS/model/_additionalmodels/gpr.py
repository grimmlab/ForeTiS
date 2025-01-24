from sklearn import gaussian_process
import sklearn
import itertools

from ForeTiS.model import _sklearn_model
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared, RBF, \
    ConstantKernel


class Gpr(_sklearn_model.SklearnModel):
    """
    Implementation of a class for Gpr.

    See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the attributes.
    """

    def define_model(self) -> gaussian_process.GaussianProcessRegressor:
        """
        Definition of the actual prediction model.

        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information.
        """
        # all hyperparameters defined for XGBoost are suggested for optimization
        self.conf = True

        self.standardize_X = self.suggest_hyperparam_to_optuna('standardize_X')
        self.standardize_y = self.suggest_hyperparam_to_optuna('normalize_y')
        if self.standardize_X:
            self.x_scaler = sklearn.preprocessing.StandardScaler()
        if self.standardize_y:
            self.y_scaler = sklearn.preprocessing.StandardScaler()

        kernel_key = self.suggest_hyperparam_to_optuna('kernel')
        kernel = self.kernel_dict[kernel_key]
        random_state = 42
        alpha = self.suggest_hyperparam_to_optuna('alpha')
        optimizer = self.suggest_hyperparam_to_optuna('optimizer')
        n_restarts_optimizer = self.suggest_hyperparam_to_optuna('n_restarts_optimizer')
        copy_X_train = self.suggest_hyperparam_to_optuna('copy_X_train')
        return gaussian_process.GaussianProcessRegressor(kernel=kernel, random_state=random_state, alpha=alpha,
                                                         optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
                                                         copy_X_train=copy_X_train)

    def define_hyperparams_to_tune(self) -> dict:
        """
        See :obj:`~ForeTiS.model._base_model.BaseModel` for more information on the format.
        """
        kernels, self.kernel_dict = self.extend_kernel_combinations()
        return {
            'kernel': {
                'datatype': 'categorical',
                'list_of_values': kernels,
            },
            'alpha': {
                'datatype': 'float',
                'lower_bound': 1e-5,
                'upper_bound': 1e3,
                'log': True
            },
            'optimizer': {
                'datatype': 'categorical',
                'list_of_values': ['fmin_l_bfgs_b']
            },
            'n_restarts_optimizer': {
                'datatype': 'int',
                'lower_bound': 0,
                'upper_bound': 100
            },
            'normalize_y': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            },
            'standardize_X': {
                'datatype': 'categorical',
                'list_of_values': [True, False]
            },
            'copy_X_train': {
                'datatype': 'categorical',
                'list_of_values': [True]
            }
        }

    def extend_kernel_combinations(self):
        """
        Function extending kernels list with combinations based on base_kernels
        """
        kernels = []
        base_kernels = ['DotProduct', 'WhiteKernel', 'Matern', 'RationalQuadratic', 'RBF', 'ConstantKernel',
                        'ExpSineSquared']
        kernel_dict = {
            'DotProduct': DotProduct(sigma_0_bounds=(1e-10, 1e5)),
            'WhiteKernel': WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5)),
            'Matern': Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
            'RationalQuadratic': RationalQuadratic(length_scale=1.0, alpha=1.0,
                                                   length_scale_bounds=(1e-5, 1e5), alpha_bounds=(1e-5, 1e5)),
            'RBF': RBF(length_scale=1.0, length_scale_bounds=(1e-5, 1e5)),
            'ConstantKernel': ConstantKernel(constant_value=1000, constant_value_bounds=(1e-5, 1e5)),
            'ExpSineSquared': ExpSineSquared(length_scale=1.0, periodicity=52, length_scale_bounds=(1e-5, 1e5),
                                             periodicity_bounds=(int(52*0.8), int(52*1.2)))
        }
        kernels.extend(base_kernels)
        for el in list(itertools.combinations(*[base_kernels], r=2)):
            kernels.append(el[0] + '+' + el[1])
            kernel_dict[el[0] + '+' + el[1]] = kernel_dict[el[0]] + kernel_dict[el[1]]
            kernels.append(el[0] + '*' + el[1])
            kernel_dict[el[0] + '*' + el[1]] = kernel_dict[el[0]] * kernel_dict[el[1]]
        for el in list(itertools.combinations(*[base_kernels], r=3)):
            kernels.append(el[0] + '+' + el[1] + '+' + el[2])
            kernel_dict[el[0] + '+' + el[1] + '+' + el[2]] = kernel_dict[el[0]] + kernel_dict[el[1]] + kernel_dict[
                el[2]]
            kernels.append(el[0] + '*' + el[1] + '*' + el[2])
            kernel_dict[el[0] + '*' + el[1] + '*' + el[2]] = kernel_dict[el[0]] * kernel_dict[el[1]] * kernel_dict[
                el[2]]
            kernels.append(el[0] + '*' + el[1] + '+' + el[2])
            kernel_dict[el[0] + '*' + el[1] + '+' + el[2]] = kernel_dict[el[0]] * kernel_dict[el[1]] + kernel_dict[
                el[2]]
            kernels.append(el[0] + '+' + el[1] + '*' + el[2])
            kernel_dict[el[0] + '+' + el[1] + '*' + el[2]] = kernel_dict[el[0]] + kernel_dict[el[1]] * kernel_dict[
                el[2]]
            kernels.append(el[0] + '*' + el[2] + '+' + el[1])
            kernel_dict[el[0] + '*' + el[2] + '+' + el[1]] = kernel_dict[el[0]] * kernel_dict[el[2]] + kernel_dict[
                el[1]]
        return kernels, kernel_dict
