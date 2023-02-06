import pprint
import configparser

from ForeTiS.utils import helper_functions
from ForeTiS.preprocess import base_dataset
from ForeTiS.optimization import optuna_optim


def run(data_dir: str, save_dir: str = None, datasplit: str = 'timeseries-cv', test_set_size_percentage=None,
        val_set_size_percentage: int = 20, imputation_method: str = 'None', windowsize_current_statistics: int = 4,
        windowsize_lagged_statistics: int = 4, models: list = None, n_trials: int = 100, pca_transform: bool =False,
        save_final_model: bool = False, periodical_refit_cycles: list = None, refit_drops: int = 0, data: str = None,
        config_file: str = None, refit_window: int = 5, intermediate_results_interval: int = None, batch_size: int = 32,
        n_epochs: int = None, event_lags: int = None, optimize_featureset: bool = None, scale_thr: float = None,
        scale_seasons: int = None, cf_thr_perc: int = None, scale_window_factor: float = None, cf_r: float = None,
        cf_order: int = None, cf_smooth: int = None, scale_window_minimum: int = None, max_samples_factor: int = None,
        valtest_seasons: int = None, seasonal_valtest: bool = None):

    # Optimization Pipeline #
    helper_functions.set_all_seeds()
    models_to_optimize = helper_functions.get_list_of_implemented_models() if models == ['all'] else models
    featureset_overview = {}
    model_featureset_overview = {}
    featureset_names = []
    config = configparser.ConfigParser(allow_no_value=True)
    config.read('Config/dataset_specific_config.ini')
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
