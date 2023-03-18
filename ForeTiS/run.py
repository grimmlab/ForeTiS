import argparse
import warnings
from optuna.exceptions import ExperimentalWarning

from ForeTiS.utils import helper_functions
from . import optim_pipeline

if __name__ == '__main__':
    """
    Run file to start the whole procedure:
            Parameter Input 
            Check and prepare data files
            Bayesian optimization for each chosen model
            Evaluation
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=ExperimentalWarning)

    # User Input #
    parser = argparse.ArgumentParser()
    # Input Params #
    parser.add_argument("-dd", "--data_dir", type=str, default='docs/source/tutorials/tutorial_data',
                        help="Provide the full path of your data directory.")
    parser.add_argument("-sd", "--save_dir", type=str, default='docs/source/tutorials/tutorial_data',
                        help="Provide the full path of the directory in which you want to save your results. "
                             "Default is same as data_dir.")
    parser.add_argument("-data", "--data", type=str, default='nike_sales',
                        help="specify the dataset that you want to use."
                        "Default: BikeSharing.")
    parser.add_argument("-con_path", "--config_file_path", type=str, default=None,
                        help="specify the path to the config file you want to use."
                             "Default: dataset_specific_config.ini in ForeTiS root folder")
    parser.add_argument("-con_sec", "--config_file_section", type=str, default='nike_sales',
                        help="specify the section within the specified config file you want to use."
                             "Default: same as name of the dataset.")
    parser.add_argument("-mod", "--models", nargs='+', default=['all'],
                        help="specify the models to optimize: 'all' or naming according to source file name. "
                             "Multiple models can be selected by just naming multiple model names, "
                             "e.g. --models mlp xgboost. "
                             "Default is 'all'"
                             "The following are available: " + str(helper_functions.get_list_of_implemented_models()))
    parser.add_argument("-of", "--optimize_featureset", type=bool, default=False,
                        help="specify whether featureset will be optimized."
                             "Standard is False")

    # Data Engineering Params
    parser.add_argument("-wf", "--windowsize_current_statistics", type=int, default=3,
                        help="specify the windowsize for the feature engineering of the current statistics. "
                             "Standard is 3")
    parser.add_argument("-ws", "--windowsize_lagged_statistics", type=int, default=3,
                        help="specify the windowsize for the feature engineering of the lagged statistics. "
                             "Standard is 3")
    parser.add_argument("-im", "--imputation_method", type=str, default='mean',
                        help="Only relevant if imputation is set in dataset_specific_config.ini: "
                             "define the imputation method to use: 'mean' | 'knn' | 'iterative'. "
                             "Standard is 'mean'")
    parser.add_argument("-ec", "--event_lags", nargs='+', default=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3],
                        help="The event lags for the counters"
                             "Standard is [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3]")

    # Preprocess Params #
    parser.add_argument("-split", "--datasplit", type=str, default='timeseries-cv',
                        help="specify the data split method to use: 'timeseries-cv' | 'train-val-test' | 'cv'. "
                             "Standard is timeseries-cv")
    parser.add_argument("-sv", "--seasonal_valtest", action='store_true', default=True,
                        help="specify whether validation and test sets should be a multiple of the season length or"
                             "a percentage of the dataset. "
                             "Either pass --seasonal-valtest for True or --no-seasonal_valtest for False.")
    parser.add_argument("-nsv", "--no-seasonal_valtest", dest='seasonal_valtest', action='store_false', default=False,
                        help="specify whether validation and test sets should be a multiple of the season length or"
                             "a percentage of the dataset. "
                             "Either pass --seasonal-valtest for True or --no-seasonal_valtest for False.")
    parser.set_defaults(seasonal_valtest=True)
    parser.add_argument("-vs", "--valtest_seasons", type=int, default=1,
                        help="Only relevant when seasonal_valtest is True: "
                             "define the number of seasons to be used. "
                             "Standard is 1")
    parser.add_argument("-testperc", "--test_set_size_percentage", type=int, default=20,
                        help="Only relevant when seasonal_valtest is False: "
                             "specify the size of the test set in percentage. "
                             "Standard is 20")
    parser.add_argument("-valperc", "--val_set_size_percentage", type=int, default=20,
                        help="Only relevant for data split methods 'train-val-test' and 'cv' "
                             "or when seasonal_valtest is False, "
                             "define the size of the validation set in percentage. "
                             "Standard is 20")
    parser.add_argument("-splits", "--n_splits", type=int, default=3,
                        help="Only relevant for datasplit method 'cv': define the number of "
                             "splits to use for 'timeseries-cv' or 'cv'. "
                             "Standard is 3")

    # Model and Optimization Params #
    parser.add_argument("-tr", "--n_trials", type=int, default=200,
                        help="specify the number of trials for the Bayesian optimization (optuna)."
                        "Standard is 200")
    parser.add_argument("-sf", "--save_final_model", type=bool, default=True,
                        help="specify whether to save the final model to hard drive or not "
                             "(caution: some models may use a lot of disk space, "
                             "unfitted models that can be retrained are already saved by default)."
                             "Standard is True")
    parser.add_argument("-prc", "--periodical_refit_frequency", type=list, default=['complete', 0, 1, 2, 3],
                        help="specify with which periods periodical refitting will be done. "
                             "0 means no periodical refitting, "
                             "complete means no periodical refitting and the whole train dataset will be used for "
                             "retraining instead of the refit_window specified below. "
                             "Standard is ['complete', 0, 1, 2, 3]")
    parser.add_argument("-rd", "--refit_drops", type=int, default=0,
                        help="specify how many rows of the train dataset get dropped before refitting. "
                             "Standard is 0")
    parser.add_argument("-rw", "--refit_window", type=int, default=5,
                        help="specify how many seasons get used for refitting. "
                             "Standard ist 5")
    parser.add_argument("-iri", "--intermediate_results_interval", type=int, default=None,
                        help="specify the number of trials after which intermediate results will be calculated. "
                             "Standard is None")
    parser.add_argument("-pca", "--pca_transform", type=bool, default=False,
                        help="specify whether pca dimensionality reduction will be performed or not. When True is passed,"
                             "it will be optimized."
                             "Standard is False")

    # Only relevant for Neural Networks #
    parser.add_argument("-bs", "--batch_size", type=int, default=None,
                        help="Only relevant for neural networks: define the batch size. If nothing is specified,"
                             "it will be considered as a hyperparameter for optimization."
                             "Standard is None")
    parser.add_argument("-ep", "--n_epochs", type=int, default=100000,
                        help="Only relevant for neural networks: define the number of epochs. If nothing is specified,"
                             "it will be considered as a hyperparameter for optimization."
                             "Standard is 100000")

    # Only relevant for EVARS-GPR(++) #
    # Pipeline
    parser.add_argument("-scalethr", "--scale_thr", type=float, default=None,
                        help="Only relevant for evars-gpr(++): specify output scale threshold. If nothing is specified,"
                             "it will be considered as a hyperparameter for optimization.")
    parser.add_argument("-scaleseas", "--scale_seasons", type=int, default=2,
                        help="Only relevant for evars-gpr(++): specify output scale seasons taken into account")
    parser.add_argument("-scalew-factor", "--scale_window_factor", type=float, default=0.1,
                        help="Only relevant for evars-gpr(++): specify scale window factor based on seasonal periods")
    parser.add_argument("-scalew-min", "--scale_window_minimum", type=int, default=2,
                        help="Only relevant for evars-gpr(++): specify scale window minimum")
    parser.add_argument("-max-samples-fact", "--max_samples_factor", type=int, default=5,
                        help="Only relevant for evars-gpr(++): specify max samples factor of seasons to keep for gpr "
                             "pipeline")
    # CF
    parser.add_argument("-cfr", "--cf_r", type=float, default=0.4,
                        help="Only relevant for evars-gpr(++): specify changefinders r param (decay factor older "
                             "values)")
    parser.add_argument("-cforder", "--cf_order", type=int, default=1,
                        help="Only relevant for evars-gpr(++): specify changefinders SDAR model order param")
    parser.add_argument("-cfsmooth", "--cf_smooth", type=int, default=4,
                        help="specify changefinders smoothing param")
    parser.add_argument("-cfthrperc", "--cf_thr_perc", type=int, default=70,
                        help="Only relevant for evars-gpr(++): specify percentile of train set anomaly factors as "
                             "threshold for cpd with changefinder")

    args = vars(parser.parse_args())

    optim_pipeline.run(**args)
