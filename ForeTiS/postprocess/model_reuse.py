import pandas as pd
import pathlib
import datetime
import configparser
import json

from ..preprocess import base_dataset
from ..utils import helper_functions
from ..model import _model_functions
from ..evaluation import eval_metrics


def apply_final_model(results_directory_model: str, old_data_dir: str, old_data: str, new_data_dir: str, new_data: str,
                      save_dir: str = None, config_file_path: pathlib.Path = None, retrain_model: bool = True):
    """
    Apply a final model on a new dataset. It will be applied to the whole dataset.
    So the main purpose of this function is, if you get new samples you want to predict on.
    If the final model was saved, this will be used for inference on the new dataset.
    Otherwise, it will be retrained on the initial dataset and then used for inference on the new dataset.

    The new dataset will be filtered for the SNP ids that the model was initially trained on.

    CAUTION: the SNPs of the old and the new dataset have to be the same!

    :param results_directory_model: directory that contains the model results that you want to use
    :param new_data_dir: directory that contains the new data
    :param old_data_dir: directory that contains the old data
    :param save_dir: directory to store the results
    :param old_data: the old dataset that you used
    :param new_data: the new dataset that you want to use
    :param config_file_path: the path to the config file you want to use
    :param retrain_model: whether to retrain the model with the whole old dataset
    """
    config_data = configparser.RawConfigParser(allow_no_value=True)
    config_data.read_file(open(save_dir + '/config_data.ini', 'r'))
    event_lags = json.loads(config_data[old_data]['event_lags'])
    windowsize_current_statistics = config_data[old_data].getint('windowsize_current_statistics')
    windowsize_lagged_statistics = config_data[old_data].getint('windowsize_lagged_statistics')
    imputation_method = config_data[old_data]['imputation_method']
    valtest_seasons = config_data[old_data].getint('valtest_seasons')
    config_file_section = config_data[old_data]['config_file_section']

    results_directory_model = pathlib.Path(results_directory_model)
    new_data_dir = pathlib.Path(new_data_dir)
    old_data_dir = pathlib.Path(old_data_dir)

    result_folder_name = results_directory_model.parts[-1]
    model_name = results_directory_model.parts[-2]
    datasplit, seasonal_valtest, val_set_size_percentage, test_set_size_percentage = \
        helper_functions.get_datasplit_config_info_for_resultfolder(resultfolder=result_folder_name)

    config = configparser.RawConfigParser(allow_no_value=True)
    config.read_file(open(config_file_path, 'r'))
    target_column = config[config_file_section]['target_column']

    # Prepare the new data
    print("Preparing the new dataset")
    new_datasets = base_dataset.Dataset(data_dir=new_data_dir, data=new_data, config_file_section=config_file_section,
                                        config=config, event_lags=event_lags,
                                        test_set_size_percentage=test_set_size_percentage,
                                        windowsize_current_statistics=windowsize_current_statistics,
                                        windowsize_lagged_statistics=windowsize_lagged_statistics,
                                        imputation_method=imputation_method, valtest_seasons=valtest_seasons,
                                        seasonal_valtest=seasonal_valtest)
    featureset_name = results_directory_model.parts[-1].split('_')[-5]
    for featureset in new_datasets.featuresets:
        if featureset.name == featureset_name:
            new_featureset = featureset
            break

    # Prepare the model
    full_model_path = results_directory_model.joinpath("final_retrained_model")
    print("Loading saved model")
    model = _model_functions.load_model(path=results_directory_model, filename=full_model_path.parts[-1])
    if retrain_model:
        print("Retraining model")
        print("Loading old dataset")
        old_datasets = base_dataset.Dataset(
            data_dir=old_data_dir, data=old_data, config_file_section=config_file_section,
            config=config, event_lags=event_lags,
            test_set_size_percentage=test_set_size_percentage,
            windowsize_current_statistics=windowsize_current_statistics,
            windowsize_lagged_statistics=windowsize_lagged_statistics,
            imputation_method=imputation_method, valtest_seasons=valtest_seasons,
            seasonal_valtest=seasonal_valtest
        )
        for featureset in old_datasets.featuresets:
            if featureset.name == featureset_name:
                old_featureset = featureset
                break
        model = _model_functions.retrain_model_with_results_file(featureset=old_featureset, model=model)

    # Do inference and save results
    print('-----------------------------------------------')
    print("Inference on new data for " + model_name)
    if hasattr(model, 'conf'):
        y_pred_new_dataset, y_pred_test_pred_var_artifical, y_pred_test_pred_conf = model.predict(X_in=new_featureset)
    else:
        y_pred_new_dataset, y_pred_test_pred_var_artifical = model.predict(X_in=new_featureset)
    eval_scores = \
        eval_metrics.get_evaluation_report(y_pred=y_pred_new_dataset, y_true=new_featureset[target_column],
                                           prefix='test_')
    print('New dataset: ')
    print(eval_scores)
    final_results = pd.DataFrame(index=range(0, new_featureset[target_column].shape[0]))
    final_results.at[0:len(y_pred_new_dataset) - 1, 'y_pred_test'] = y_pred_new_dataset.flatten()
    final_results.at[0:len(new_featureset[target_column]) - 1, 'y_true_test'] = new_featureset[
        target_column].values.reshape(-1)
    for metric, value in eval_scores.items():
        final_results.at[0, metric] = value
    final_results.at[0, 'base_model_path'] = results_directory_model
    models_start_time = model_name + '_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_results.to_csv(results_directory_model.joinpath(
        'predict_results_on-' + new_featureset.name + '-' + models_start_time + '.csv'),
        sep=',', decimal='.', float_format='%.10f', index=False
    )
