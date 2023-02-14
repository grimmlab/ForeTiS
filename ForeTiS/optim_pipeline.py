import pprint
import configparser
import pathlib
from pathlib import Path

from ForeTiS.utils import helper_functions
from ForeTiS.preprocess import base_dataset
from ForeTiS.optimization import optuna_optim


def run(data_dir: str, save_dir: str, datasplit: str = 'timeseries-cv', test_set_size_percentage: int = 20,
        val_set_size_percentage: int = 20, imputation_method: str = None, windowsize_current_statistics: int = 3,
        windowsize_lagged_statistics: int = 3, models: list = None, n_trials: int = 200, pca_transform: bool =False,
        save_final_model: bool = False, periodical_refit_cycles: list = None, refit_drops: int = 0, data: str = None,
        config_file: str = None, refit_window: int = 5, intermediate_results_interval: int = None, batch_size: int = 32,
        n_epochs: int = 100000, event_lags: int = None, optimize_featureset: bool = False, scale_thr: float = 0.1,
        scale_seasons: int = 2, cf_thr_perc: int = 70, scale_window_factor: float = 0.1, cf_r: float = 0.4,
        cf_order: int = 1, cf_smooth: int = 4, scale_window_minimum: int = 2, max_samples_factor: int = 10,
        valtest_seasons: int = 1, seasonal_valtest: bool = True):

    # create Path
    data_dir = pathlib.Path(data_dir)
    # set save directory
    save_dir = data_dir if save_dir is None else pathlib.Path(save_dir)
    save_dir = save_dir if save_dir.is_absolute() else save_dir.resolve()
    # Optimization Pipeline #
    helper_functions.set_all_seeds()
    models_to_optimize = helper_functions.get_list_of_implemented_models() if models == ['all'] else models
    featureset_overview = {}
    model_featureset_overview = {}
    featureset_names = []
    config = configparser.ConfigParser(allow_no_value=True)
    root = Path(__file__).parent
    ini_path = root.joinpath('dataset_specific_config.ini')
    config.read_file(open(ini_path, 'r'))
    datasets = base_dataset.Dataset(data_dir=data_dir, data=data, config_file=config_file, event_lags=event_lags,
                                    test_set_size_percentage=test_set_size_percentage,
                                    windowsize_current_statistics=windowsize_current_statistics,
                                    windowsize_lagged_statistics=windowsize_lagged_statistics,
                                    imputation_method=imputation_method, config=config, valtest_seasons=valtest_seasons,
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
            optuna_run = optuna_optim.OptunaOptim(save_dir=save_dir, data=data, config_type=config_file,
                                                  featureset_name=featureset_name, datasplit=datasplit,
                                                  n_trials=n_trials, test_set_size_percentage=test_set_size_percentage,
                                                  models=models, val_set_size_percentage=val_set_size_percentage,
                                                  save_final_model=save_final_model, pca_transform=pca_transform,
                                                  periodical_refit_cycles=periodical_refit_cycles,
                                                  refit_drops=refit_drops, refit_window=refit_window,
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
