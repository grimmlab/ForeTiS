import pprint
import configparser
import pathlib
from pathlib import Path

from ForeTiS.utils import helper_functions
from ForeTiS.preprocess import base_dataset
from ForeTiS.optimization import optuna_optim


def run(data_dir: str, save_dir: str, datasplit: str = 'timeseries-cv', test_set_size_percentage: int = 20,
        val_set_size_percentage: int = 20, n_splits: int = 3, imputation_method: str = None,
        windowsize_current_statistics: int = 3, windowsize_lagged_statistics: int = 3, models: list = None,
        n_trials: int = 200, pca_transform: bool =False, save_final_model: bool = False,
        periodical_refit_frequency: list = None, refit_drops: int = 0, data: str = None, config_file_path: str = None,
        config_file_section: str = None, refit_window: int = 5, intermediate_results_interval: int = None,
        batch_size: int = 32, n_epochs: int = 100000, event_lags: int = None, optimize_featureset: bool = False,
        scale_thr: float = 0.1, scale_seasons: int = 2, cf_thr_perc: int = 70, scale_window_factor: float = 0.1,
        cf_r: float = 0.4, cf_order: int = 1, cf_smooth: int = 4, scale_window_minimum: int = 2,
        max_samples_factor: int = 10, valtest_seasons: int = 1, seasonal_valtest: bool = True):
    """
    Run the whole optimization pipeline

    :param data_dir: data directory where the phenotype and genotype matrix are stored
    :param save_dir: directory for saving the results. Default is None, so same directory as data_dir
    :param datasplit: datasplit to use. Options are: nested-cv, cv-test, train-val-test
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    :param n_splits: splits to use for 'timeseries-cv' or 'cv'
    :param imputation_method: the imputation method to use. Options are: 'mean' , 'knn' , 'iterative'
    :param windowsize_current_statistics: the windowsize for the feature engineering of the current statistic
    :param windowsize_lagged_statistics: the windowsize for the feature engineering of the lagged statistics
    :param models: list of models that should be optimized
    :param n_trials: number of trials for optuna
    :param pca_transform: whether pca dimensionality reduction will be optimized or not
    :param save_final_model: specify if the final model should be saved
    :param periodical_refit_frequency: if and for which intervals periodical refitting should be performed
    :param refit_drops: after how many periods the model should get updated
    :param data: the dataset that you want to use
    :param config_file_path: the path of the config file
    :param config_file_section: the section of the config file for the used dataset
    :param refit_window: seasons get used for refitting
    :param intermediate_results_interval: number of trials after which intermediate results will be saved
    :param batch_size: batch size for neural network models
    :param n_epochs: number of epochs for neural network models
    :param event_lags: the event lags for the counters
    :param optimize_featureset: whether feature set will be optimized or not output scale threshold
    :param scale_thr: only relevant for evars-gpr: output scale threshold
    :param scale_seasons: only relevant for evars-gpr: output scale seasons taken into account
    :param cf_thr_perc: only relevant for evars-gpr: percentile of train set anomaly factors as threshold for cpd with changefinder
    :param scale_window_factor: only relevant for evars-gpr: scale window factor based on seasonal periods
    :param cf_r: only relevant for evars-gpr: changefinders r param (decay factor older values)
    :param cf_order: only relevant for evars-gpr: changefinders SDAR model order param
    :param cf_smooth: only relevant for evars-gpr: changefinders smoothing param
    :param scale_window_minimum: only relevant for evars-gpr: scale window minimum
    :param max_samples_factor: only relevant for evars-gpr: max samples factor of seasons to keep for gpr pipeline
    :param valtest_seasons: define the number of seasons to be used when seasonal_valtest is True
    :param seasonal_valtest: whether validation and test sets should be a multiple of the season length
    """
    # create Path
    data_dir = pathlib.Path(data_dir)
    config_file_path = pathlib.Path(config_file_path) if config_file_path is not None else \
        Path(__file__).parent.joinpath('dataset_specific_config.ini')
    # set save directory
    save_dir = data_dir if save_dir is None else pathlib.Path(save_dir)
    save_dir = save_dir if save_dir.is_absolute() else save_dir.resolve()
    # set config_file_section to dataset name if it is none
    config_file_section = config_file_section if config_file_section is not None else data
    # Optimization Pipeline #
    helper_functions.set_all_seeds()
    models_to_optimize = helper_functions.get_list_of_implemented_models() if models == ['all'] else models
    featureset_overview = {}
    model_featureset_overview = {}
    featureset_names = []
    config = configparser.ConfigParser(allow_no_value=True)
    config.read_file(open(config_file_path, 'r'))
    datasets = base_dataset.Dataset(data_dir=data_dir, data=data, config_file_section=config_file_section, config=config,
                                    event_lags=event_lags, test_set_size_percentage=test_set_size_percentage,
                                    windowsize_current_statistics=windowsize_current_statistics,
                                    windowsize_lagged_statistics=windowsize_lagged_statistics,
                                    imputation_method=imputation_method, valtest_seasons=valtest_seasons,
                                    seasonal_valtest=seasonal_valtest)
    print('### Dataset is loaded ###')
    for current_model_name in models_to_optimize:
        if optimize_featureset:
            featuresets_list = ['optimize']
        else:
            featuresets_list = datasets.featuresets
        for featureset in featuresets_list:
            if optimize_featureset:
                featureset_name = "optimize"
            else:
                featureset_name = featureset.name
            featureset_names.append(featureset_name)
            optuna_run = optuna_optim.OptunaOptim(save_dir=save_dir, data=data, config_file_section=config_file_section,
                                                  featureset_name=featureset_name, datasplit=datasplit,
                                                  n_trials=n_trials, test_set_size_percentage=test_set_size_percentage,
                                                  val_set_size_percentage=val_set_size_percentage,
                                                  save_final_model=save_final_model, pca_transform=pca_transform,
                                                  periodical_refit_frequency=periodical_refit_frequency,
                                                  refit_drops=refit_drops, refit_window=refit_window, n_splits=n_splits,
                                                  intermediate_results_interval=intermediate_results_interval,
                                                  batch_size=batch_size, n_epochs=n_epochs, datasets=datasets,
                                                  current_model_name=current_model_name, config=config,
                                                  optimize_featureset=optimize_featureset, scale_thr=scale_thr,
                                                  scale_seasons=scale_seasons, scale_window_factor=scale_window_factor,
                                                  cf_r=cf_r, cf_order=cf_order, cf_smooth=cf_smooth,
                                                  cf_thr_perc=cf_thr_perc, scale_window_minimum=scale_window_minimum,
                                                  max_samples_factor=max_samples_factor, valtest_seasons=valtest_seasons,
                                                  seasonal_valtest=seasonal_valtest)
            print('### Starting Optuna Optimization for model ' + current_model_name + ' and featureset ' +
                  featureset_name + ' ###')
            overall_results = optuna_run.run_optuna_optimization
            print('### Finished Optuna Optimization for ' + current_model_name + ' and featureset ' + featureset_name
                  + ' ###')
            featureset_overview[featureset_name] = overall_results
            model_featureset_overview[current_model_name] = featureset_overview
    print('# Optimization runs done for models ' + str(models_to_optimize) + ' and ' + str(featureset_names))
    print('Results overview on the test set(s)')
    pprint.PrettyPrinter(depth=5).pprint(model_featureset_overview)
