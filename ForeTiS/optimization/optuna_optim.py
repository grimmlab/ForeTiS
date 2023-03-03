import datetime
import optuna
import pandas as pd
import sklearn
import numpy as np
import os
import warnings
import shutil
from sklearn.model_selection import train_test_split
import csv
import time
import traceback
import copy
import configparser
import matplotlib.pyplot as plt
import matplotlib as mpl
import pathlib

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.xmargin'] = 0.015
plt.style.use('ggplot')


from ..preprocess import base_dataset
from ..utils import helper_functions
from ..evaluation import eval_metrics
from ..model import _base_model, _model_functions, _torch_model, _stat_model


class OptunaOptim:
    """
    Class that contains all info for the whole optimization using optuna for one model and dataset.

    ** Attributes **

        - study (*optuna.study.Study*): optuna study for optimization run
        - current_best_val_result (*float*): the best validation result so far
        - early_stopping_point (*int*): point at which early stopping occured (relevant for some models)
        - seasonal_periods (*int*): number of samples in one season of the used dataset
        - target_column (*str*): target column for which predictions shall be made
        - best_trials (*list*): list containing the numbers of the best trials
        - user_input_params (*dict*): all params handed over to the constructor that are needed in the whole class
        - base_path (*str*): base_path for save_path
        - save_path (*str*): path for model and results storing

    :param save_dir: directory for saving the results
    :param data: the dataset that you want to use
    :param config_file_section: the section of the config file for the used dataset
    :param featureset_name: name of the feature set used
    :param datasplit: the used datasplit method, either 'timeseries-cv', 'train-val-test', 'cv'
    :param test_set_size_percentage: size of the test set relevant for cv-test and train-val-test
    :param val_set_size_percentage: size of the validation set relevant for train-val-test
    :param n_trials: number of trials for optuna
    :param save_final_model: specify if the final model should be saved
    :param batch_size: batch size for neural network models
    :param n_epochs: number of epochs for neural network models
    :param current_model_name: name of the current model according to naming of .py file in package model
    :param datasets: the Dataset class containing the feature sets
    :param periodical_refit_frequency: if and for which intervals periodical refitting should be performed
    :param refit_drops: after how many periods the model should get updated
    :param refit_window: seasons get used for refitting
    :param intermediate_results_interval: number of trials after which intermediate results will be saved
    :param pca_transform: whether pca dimensionality reduction will be optimized or not
    :param config: the information from dataset_specific_config.ini
    :param optimize_featureset: whether feature set will be optimized or not output scale threshold
    :param scale_thr: only relevant for evars-gpr: output scale threshold
    :param scale_seasons: only relevant for evars-gpr: output scale seasons taken into account
    :param scale_window_factor: only relevant for evars-gpr: scale window factor based on seasonal periods
    :param cf_r: only relevant for evars-gpr: changefinders r param (decay factor older values)
    :param cf_order: only relevant for evars-gpr: changefinders SDAR model order param
    :param cf_smooth: only relevant for evars-gpr: changefinders smoothing param
    :param cf_thr_perc: only relevant for evars-gpr: percentile of train set anomaly factors as threshold for cpd with changefinder
    :param scale_window_minimum: only relevant for evars-gpr: scale window minimum
    :param max_samples_factor: only relevant for evars-gpr: max samples factor of seasons to keep for gpr pipeline
    :param valtest_seasons: define the number of seasons to be used when seasonal_valtest is True
    :param seasonal_valtest: whether validation and test sets should be a multiple of the season length
    :param n_splits: splits to use for 'timeseries-cv' or 'cv'
    """

    def __init__(self, save_dir: pathlib.Path, data: str, config_file_section: str, featureset_name: str,
                 datasplit: str, test_set_size_percentage: int, val_set_size_percentage: int, n_trials: int,
                 save_final_model: bool, batch_size: int, n_epochs: int, current_model_name: str,
                 datasets: base_dataset.Dataset, periodical_refit_frequency: list, refit_drops: int, refit_window: int,
                 intermediate_results_interval: int, pca_transform: bool, config: configparser.ConfigParser,
                 optimize_featureset: bool, scale_thr: float, scale_seasons: int, scale_window_factor: float, cf_r: float,
                 cf_order: int, cf_smooth: int, cf_thr_perc: int, scale_window_minimum: int, max_samples_factor: int,
                 valtest_seasons: int, seasonal_valtest: bool, n_splits: int):
        self.study = None
        self.current_best_val_result = None
        self.early_stopping_point = None
        self.seasonal_periods = config[config_file_section].getint('seasonal_periods')
        self.target_column = config[config_file_section]['target_column']
        self.best_trials = []
        self.user_input_params = locals()  # distribute all handed over params in whole class
        self.base_path = save_dir.joinpath('results', current_model_name,
                                           datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' +
                                           self.user_input_params['featureset_name'])
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        self.save_path = self.base_path

        if self.user_input_params['seasonal_valtest'] and \
                (self.user_input_params['datasplit'] == 'cv' or self.user_input_params['datasplit'] == 'train-val-test'):
            warnings.warn('seasonal_valtest is set to true, while datasplit is set to cv or train-val-test. '
                          'Will ignore seasonal_valtest.')

    def create_new_study(self) -> optuna.study.Study:
        """
        Create a new optuna study.

        :return: a new optuna study instance
        """
        study_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + '-MODEL' + \
                     self.user_input_params['current_model_name'] + '-TRIALS' + str(self.user_input_params["n_trials"]) \
                     + '-FEATURESET' + self.user_input_params['featureset_name']
        storage = optuna.storages.RDBStorage(
            "sqlite:///" + str(self.save_path.joinpath('Optuna_DB.db')), heartbeat_interval=60, grace_period=120,
            failed_trial_callback=optuna.storages.RetryFailedTrialCallback(max_retry=3))

        study = optuna.create_study(
            storage=storage, study_name=study_name, direction='minimize', sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.PercentilePruner(percentile=80, n_min_trials=20), load_if_exists=True)

        return study

    def objective(self, trial: optuna.trial.Trial):
        """
        Objective function for optuna optimization that returns a score

        :param trial: trial of optuna for optimization

        :return: score of the current hyperparameter config
        """
        if (trial.number != 0) and (self.user_input_params["intermediate_results_interval"] is not None) and (
                trial.number % self.user_input_params["intermediate_results_interval"] == 0):
            print('Generate intermediate test results at trial ' + str(trial.number))
            _ = self.generate_results_on_test()

        # Setup timers for runtime logging
        start_process_time = time.process_time()
        start_realclock_time = time.time()

        # Create model
        # in case a model has attributes not part of the base class hand them over in a dictionary to keep the same call
        # (name of the attribute and key in the dictionary have to match)
        additional_attributes_dict = {}
        if issubclass(helper_functions.get_mapping_name_to_class()[self.user_input_params['current_model_name']],
                      _torch_model.TorchModel) \
                or issubclass(helper_functions.get_mapping_name_to_class()[self.user_input_params['current_model_name']],
                              _stat_model.StatModel):
            # additional attributes for torch and stats models
            additional_attributes_dict['current_model_name'] = self.user_input_params['current_model_name']
            if issubclass(helper_functions.get_mapping_name_to_class()[self.user_input_params['current_model_name']],
                          _torch_model.TorchModel):
                additional_attributes_dict['batch_size'] = self.user_input_params["batch_size"]
                additional_attributes_dict['n_epochs'] = self.user_input_params["n_epochs"]
                early_stopping_points = []  # log early stopping point at each fold for torch models
        if self.user_input_params['current_model_name'] == 'evars-gpr':
            additional_attributes_dict['scale_thr'] = self.user_input_params["scale_thr"]
            additional_attributes_dict['scale_seasons'] = self.user_input_params["scale_seasons"]
            additional_attributes_dict['scale_window_factor'] = self.user_input_params["scale_window_factor"]
            additional_attributes_dict['scale_window_minimum'] = self.user_input_params["scale_window_minimum"]
            additional_attributes_dict['max_samples_factor'] = self.user_input_params["max_samples_factor"]
            additional_attributes_dict['cf_r'] = self.user_input_params["cf_r"]
            additional_attributes_dict['cf_order'] = self.user_input_params["cf_order"]
            additional_attributes_dict['cf_smooth'] = self.user_input_params["cf_smooth"]
            additional_attributes_dict['cf_thr_perc'] = self.user_input_params["cf_thr_perc"]
            additional_attributes_dict['scale_window_minimum'] = self.user_input_params["scale_window_minimum"]
            additional_attributes_dict['max_samples_factor'] = self.user_input_params["max_samples_factor"]
        try:
            model: _base_model.BaseModel = \
                helper_functions.get_mapping_name_to_class()[self.user_input_params['current_model_name']]\
                    (target_column=self.target_column, datasets=self.user_input_params['datasets'],
                     featureset_name=self.user_input_params['featureset_name'], optuna_trial=trial,
                     pca_transform=self.user_input_params['pca_transform'],
                     optimize_featureset=self.user_input_params['optimize_featureset'], **additional_attributes_dict)
        except Exception as exc:
            print(traceback.format_exc())
            print(exc)
            print(trial.params)
            print('Trial failed. Error in model creation.')
            self.clean_up_after_exception(
                trial_number=trial.number, trial_params=trial.params, reason='model creation: ' + str(exc))
            raise optuna.exceptions.TrialPruned()

        # set the datasplit
        self.featureset = model.featureset
        if self.user_input_params['datasplit'] == 'timeseries-cv' and self.user_input_params['seasonal_valtest']:
            train_val = self.featureset.iloc[: -self.user_input_params["valtest_seasons"]*self.seasonal_periods]
            train_val.index.freq = train_val.index.inferred_freq
        else:
            train_val, _ = train_test_split(
                self.featureset, test_size=self.user_input_params["test_set_size_percentage"] * 0.01, shuffle=False)

        # Security mechanisms
        if self.user_input_params['datasplit'] != 'timeseries-cv' and self.user_input_params['current_model_name'] in ['lstm', 'lstmbayes', 'es', 'arima', 'arimax']:
            raise Exception("Model with time dependency only work together with timerseries-cv.")
        if not all(elem == "complete" for elem in self.user_input_params['periodical_refit_frequency']) and max((i for i in self.user_input_params['periodical_refit_frequency'] if isinstance(i, int))) >= (len(self.featureset) - len(train_val))//2:
            print("One or more refitting cycles are longer than the test set. Please reset the refitting cycles.")
            refitting_cycles_lst = []
            number_refit_cycles = int(input("Number of refitting cycles: "))
            for i in range(number_refit_cycles):
                refit_cycle = input('Refitting cycle number %i: ' %i)
                if refit_cycle != 'complete':
                    refit_cycle = int(refit_cycle)
                refitting_cycles_lst.append(refit_cycle)
            self.user_input_params['periodical_refit_frequency'] = refitting_cycles_lst

        # save the unfitted model
        self.save_path.joinpath('temp').mkdir(parents=True, exist_ok=True)
        model.save_model(path=self.save_path.joinpath('temp'),
                         filename='unfitted_model_trial' + str(trial.number))
        print('Params for Trial ' + str(trial.number))
        print(trial.params)
        if self.check_params_for_duplicate(current_params=trial.params):
            print('Trial params are a duplicate.')
            self.clean_up_after_exception(
                trial_number=trial.number, trial_params=trial.params, reason='pruned: duplicate')
            raise optuna.exceptions.TrialPruned()

        objective_values = []
        validation_results = pd.DataFrame(index=range(0, self.featureset.shape[0]))

        if 'cv' in self.user_input_params['datasplit']:
            if self.user_input_params['seasonal_valtest']:
                folds = round((len(train_val)/self.seasonal_periods))//2
                if folds == 0:
                    raise Exception(
                        "Can not create a single fold. Probably training set too short and seasonal periods too long.")
            else:
                folds = self.user_input_params['n_splits']
            if folds > 3:
                folds = 3
            train_indexes, val_indexes = helper_functions.get_indexes(
                df=train_val, datasplit=self.user_input_params['datasplit'],
                seasonal_valtest=self.user_input_params['seasonal_valtest'],
                valtest_seasons=self.user_input_params['valtest_seasons'], folds=folds,
                seasonal_periods=self.seasonal_periods,
                val_set_size_percentage=self.user_input_params["val_set_size_percentage"])
        else:
            folds = 1
        for fold in range(folds):
            fold_name = "fold_" + str(fold)
            if self.user_input_params['datasplit'] == "timeseries-cv" or self.user_input_params['datasplit'] == "cv":
                train, val = train_val.iloc[train_indexes[fold]], train_val.iloc[val_indexes[fold]]
            else:
                train, val = train_test_split(
                    train_val, test_size=self.user_input_params["val_set_size_percentage"] * 0.01, shuffle=False)

            if model.pca_transform:
                train, val = self.pca_transform_train_test(train, val)

            # load the unfitted model to prevent information leak between folds
            model = _model_functions.load_model(path=self.save_path.joinpath('temp'),
                                                filename='unfitted_model_trial' + str(trial.number))

            try:
                # run train and validation loop for this fold
                y_pred = model.train_val_loop(train=train, val=val)[0]

                if hasattr(model, 'early_stopping_point'):
                    early_stopping_points.append(
                        model.early_stopping_point if model.early_stopping_point is not None else model.n_epochs)

                if len(y_pred) == (len(val) - 1):
                    # might happen if batch size leads to a last batch with one sample which will be dropped then
                    print('val has one element less than y_true (e.g. due to batch size) -> drop last element')
                    val = val[:-1]

                objective_value = \
                    sklearn.metrics.mean_squared_error(y_true=val[self.target_column], y_pred=y_pred)

                # report value for pruning
                trial.report(value=objective_value,
                             step=0 if self.user_input_params['datasplit'] == 'train-val-test' else int(fold_name[-1]))
                if trial.should_prune():
                    self.clean_up_after_exception(trial_number=trial.number, trial_params=trial.params, reason='pruned')
                    raise optuna.exceptions.TrialPruned()

                # store results
                objective_values.append(objective_value)
                validation_results.at[0:len(train) - 1, fold_name + '_train_true']\
                    = train.loc[:, [self.target_column]].values.reshape(-1)
                if 'lstm' in self.user_input_params['current_model_name']:
                    try:
                        validation_results.at[0:len(train) - model.seq_length - 1, fold_name + '_train_pred'] = \
                            model.predict(X_in=train)[0]
                    except:
                        # model.predict(X_in=train)[0] has one element less than train (e.g. due to batch size)
                        validation_results.at[0:len(train) - model.seq_length - 2, fold_name + '_train_pred'] = \
                            model.predict(X_in=train)[0]
                else:
                    try:
                        validation_results.at[0:len(train) - 1, fold_name + '_train_pred'] = \
                            model.predict(X_in=train)[0]
                    except:
                        # model.predict(X_in=train)[0] has one element less than train (e.g. due to batch size)
                        validation_results.at[0:len(train) - 2, fold_name + '_train_pred'] = \
                            model.predict(X_in=train)[0]

                validation_results.at[0:len(val) - 1, fold_name + '_val_true'] = \
                    val.loc[:, [self.target_column]].values.reshape(-1)
                validation_results.at[0:len(y_pred) - 1, fold_name + '_val_pred'] = y_pred

                for metric, value in eval_metrics.get_evaluation_report(y_pred=y_pred,
                                                                        y_true=val[self.target_column],
                                                                        prefix=fold_name + '_').items():
                    validation_results.at[0, metric] = value

            except (RuntimeError, TypeError, ValueError, IndexError, NameError, np.linalg.LinAlgError) as exc:
                print(traceback.format_exc())
                print(exc)
                print('Trial failed. Error in optim loop.')
                self.clean_up_after_exception(trial_number=trial.number, trial_params=trial.params,
                                              reason='model optimization: ' + str(exc))
                raise optuna.exceptions.TrialPruned()

        current_val_result = float(np.mean(objective_values))
        if self.current_best_val_result is None or current_val_result < self.current_best_val_result:
            if hasattr(model, 'early_stopping_point'):
                # take mean of early stopping points of all folds for refitting of final model
                self.early_stopping_point = int(np.mean(early_stopping_points))
            self.current_best_val_result = current_val_result
            # persist results
            validation_results.to_csv(
                self.save_path.joinpath('temp', 'validation_results_trial' + str(trial.number) + '.csv'),
                sep=',', decimal='.', float_format='%.10f', index=False
            )
            self.best_trials.insert(0, trial.number)
        else:
            # delete unfitted model
            self.save_path.joinpath('temp', 'unfitted_model_trial' + str(trial.number)).unlink()

        # save runtime information of this trial
        self.write_runtime_csv(
            dict_runtime={'Trial': trial.number, 'process_time_s': time.process_time() - start_process_time,
                          'real_time_s': time.time() - start_realclock_time, 'params': trial.params,
                          'note': 'successful'})

        return current_val_result

    def clean_up_after_exception(self, trial_number: int, trial_params: dict, reason: str):
        """
        Clean up things after an exception: delete unfitted model if it exists and update runtime csv

        :param trial_number: number of the trial
        :param trial_params: parameters of the trial
        :param reason: hint for the reason of the Exception
        """
        if self.save_path.joinpath('temp', 'unfitted_model_trial' + str(trial_number)).exists():
            self.save_path.joinpath('temp', 'unfitted_model_trial' + str(trial_number)).unlink()
        self.write_runtime_csv(dict_runtime={'Trial': trial_number, 'process_time_s': np.nan, 'real_time_s': np.nan,
                                             'params': trial_params, 'note': reason})

    def write_runtime_csv(self, dict_runtime: dict):
        """
        Write runtime info to runtime csv file

        :param dict_runtime: dictionary with runtime information
        """
        with open(self.save_path.joinpath(self.user_input_params['current_model_name'] + '_runtime_overview.csv'), 'a') as runtime_file:
            headers = ['Trial', 'refitting_cycle', 'process_time_s', 'real_time_s', 'params', 'note']
            writer = csv.DictWriter(f=runtime_file, fieldnames=headers)
            if runtime_file.tell() == 0:
                writer.writeheader()
            writer.writerow(dict_runtime)

    def calc_runtime_stats(self) -> dict:
        """
        Calculate runtime stats for saved csv file.

        :return: dict with runtime info enhanced with runtime stats
        """
        csv_file = pd.read_csv(self.save_path.joinpath(self.user_input_params['current_model_name'] + '_runtime_overview.csv'))
        if csv_file['Trial'].dtype is object and any(["retrain" in elem for elem in csv_file["Trial"]]):
            csv_file = csv_file[csv_file["Trial"].str.contains("retrain") is False]
        process_times = csv_file['process_time_s']
        real_times = csv_file['real_time_s']
        process_time_mean, process_time_std, process_time_max, process_time_min = \
            process_times.mean(), process_times.std(), process_times.max(), process_times.min()
        real_time_mean, real_time_std, real_time_max, real_time_min = \
            real_times.mean(), real_times.std(), real_times.max(), real_times.min()
        self.write_runtime_csv({'Trial': 'mean', 'process_time_s': process_time_mean, 'real_time_s': real_time_mean})
        self.write_runtime_csv({'Trial': 'std', 'process_time_s': process_time_std, 'real_time_s': real_time_std})
        self.write_runtime_csv({'Trial': 'max', 'process_time_s': process_time_max, 'real_time_s': real_time_max})
        self.write_runtime_csv({'Trial': 'min', 'process_time_s': process_time_min, 'real_time_s': real_time_min})
        return {'process_time_mean': process_time_mean, 'process_time_std': process_time_std,
                'process_time_max': process_time_max, 'process_time_min': process_time_min,
                'real_time_mean': real_time_mean, 'real_time_std': real_time_std,
                'real_time_max': real_time_max, 'real_time_min': real_time_min}

    def check_params_for_duplicate(self, current_params: dict) -> bool:
        """
        Check if params were already suggested which might happen by design of TPE sampler.

        :param current_params: dictionar with current parameters

        :return: bool reflecting if current params were already used in the same study
        """
        past_params = [trial.params for trial in self.study.trials[:-1]]
        return current_params in past_params

    def pca_transform_train_test(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        """
        Deliver PCA transformed train and test set

        :param train: data for the training
        :param test: data for the testing

        :return: tuple of transformed train and test dataset
        """
        scaler = sklearn.preprocessing.StandardScaler()
        train_stand = scaler.fit_transform(train.drop(self.target_column, axis=1))
        pca = sklearn.decomposition.PCA(0.95)
        train_transf = pca.fit_transform(train_stand)
        test_stand = scaler.transform(test.drop(self.target_column, axis=1))
        test_transf = pca.transform(test_stand)
        train_data = pd.DataFrame(data=train_transf,
                                  columns=['PC' + str(i) for i in range(train_transf.shape[1])],
                                  index=train.index)
        train_data[self.target_column] = train[self.target_column]
        test_data = pd.DataFrame(data=test_transf, columns=['PC' + str(i) for i in range(test_transf.shape[1])],
                                 index=test.index)
        test_data[self.target_column] = test[self.target_column]
        return train_data, test_data

    def load_retrain_model(self, path: str, filename: str, retrain: pd.DataFrame, early_stopping_point: int = None,
                           test: pd.DataFrame = None) -> tuple:
        """
        Load and retrain persisted model
        :param path: path where the model is saved
        :param filename: filename of the model
        :param retrain: data for retraining
        :param test: data for testing
        :param early_stopping_point: optional early stopping point relevant for some models
        :return: model instance
        """
        model = _model_functions.load_model(path=path, filename=filename)
        if early_stopping_point is not None:
            model.early_stopping_point = early_stopping_point
        model.prediction = None
        if model.pca_transform:
            retrain, _ = self.pca_transform_train_test(retrain, test)
        model.retrain(retrain=retrain)
        return model

    def generate_results_on_test(self) -> dict:
        """
        Generate the results on the testing data

        :return: evaluation metrics dictionary
        """
        helper_functions.set_all_seeds()

        print("## Retrain best model and test ##")
        # Retrain on full train + val data with best hyperparams and apply on test
        prefix = '' if len(self.study.trials) == self.user_input_params["n_trials"] else '/temp/'
        if self.user_input_params['datasplit'] == 'timeseries-cv' and self.user_input_params['seasonal_valtest']:
            test = self.featureset.iloc[-(self.user_input_params["valtest_seasons"]*self.seasonal_periods):]
            retrain = self.featureset.iloc[:-self.user_input_params["valtest_seasons"]*self.seasonal_periods]
        else:
            retrain, test = train_test_split(
                self.featureset, test_size=self.user_input_params["test_set_size_percentage"] * 0.01, shuffle=False)

        start_process_time = time.process_time()
        start_realclock_time = time.time()
        postfix = '' if len(self.study.trials) == self.user_input_params["n_trials"] else 'temp'
        final_model = self.load_retrain_model(
            path=self.save_path.joinpath(postfix), filename=prefix + 'unfitted_model_trial' + str(self.study.best_trial_copy.number),
            retrain=retrain, test=test, early_stopping_point=self.early_stopping_point)

        if final_model.pca_transform:
            retrain, test = self.pca_transform_train_test(retrain, test)

        y_pred_retrain = final_model.predict(X_in=retrain)[0]
        final_model.var_artifical = np.quantile(
            retrain[self.target_column][-len(final_model.prediction):] - y_pred_retrain, 0.68) ** 2
        final_results = pd.DataFrame(index=range(0, self.featureset.shape[0]))

        final_results.at[0:len(y_pred_retrain) - 1, 'y_pred_retrain'] = y_pred_retrain
        final_results.at[0:len(retrain) - 1, 'y_true_retrain'] = retrain[self.target_column].values.flatten()
        final_results.at[0:len(test) - 1, 'y_true_test'] = test[self.target_column].values.flatten()

        if self.user_input_params['current_model_name'] in ['ard', 'bayesridge', 'elasticnet', 'lasso', 'ridge', 'xgboost']:
            feature_importance = pd.DataFrame(index=range(0, 0))

        for count, period in enumerate(self.user_input_params['periodical_refit_frequency']):
            test_len = test.shape[0]
            if hasattr(final_model, 'sequential'):
                test = self.featureset.tail(len(test) + final_model.seq_length)

            model = copy.deepcopy(final_model)

            if period == 'complete':
                if hasattr(final_model, 'conf'):
                    y_pred_test, y_pred_test_var, y_pred_test_conf = model.predict(X_in=test)
                else:
                    y_pred_test, y_pred_test_var = model.predict(X_in=test)
                y_pred_test_var = np.full((len(y_pred_test),), y_pred_test_var)

            elif period == 0:
                model.retrain(retrain=retrain.tail(self.user_input_params['refit_window']*self.seasonal_periods))

                if hasattr(final_model, 'conf'):
                    y_pred_test, y_pred_test_var, y_pred_test_conf = model.predict(X_in=test)
                else:
                    y_pred_test, y_pred_test_var = model.predict(X_in=test)
                y_pred_test_var = np.full((len(y_pred_test),), y_pred_test_var)

            else:
                model.retrain(retrain=retrain.tail(self.user_input_params['refit_window'] * self.seasonal_periods))
                y_pred_test = list()
                y_pred_test_var = list()
                if hasattr(final_model, 'conf'):
                    y_pred_test_conf = list()

                X_train_val_manip = retrain.tail(self.user_input_params['refit_window']*self.seasonal_periods).copy()
                X_test_manip = test.copy()
                if hasattr(model, 'sequential'):
                    x_test = model.X_scaler.transform(X_test_manip.drop(labels=[self.target_column], axis=1))
                    y_test = model.y_scaler.transform(X_test_manip[self.target_column].values.reshape(-1, 1))
                    x_test, _ = model.create_sequences(x_test, y_test)

                for i in range(test_len):
                    if hasattr(model, 'sequential'):
                        if hasattr(final_model, 'conf'):
                            y_pred_test_pred, y_pred_test_pred_var_artifical, y_pred_test_pred_conf \
                                = model.predict(X_in=x_test[0])
                            y_pred_test_conf.append(y_pred_test_pred_conf)
                        else:
                            y_pred_test_pred, y_pred_test_pred_var_artifical = model.predict(X_in=x_test[0])
                        y_pred_test.append(y_pred_test_pred)
                        y_pred_test_var.append(y_pred_test_pred_var_artifical)
                    else:
                        if hasattr(final_model, 'conf'):
                            y_pred_test_pred, y_pred_test_pred_var_artifical, y_pred_test_pred_conf \
                                = model.predict(X_in=X_test_manip.iloc[[0]])
                            y_pred_test_conf.extend(y_pred_test_pred_conf)
                        else:
                            y_pred_test_pred, y_pred_test_pred_var_artifical = \
                                model.predict(X_in=X_test_manip.iloc[[0]])
                        y_pred_test.append(y_pred_test_pred)
                        y_pred_test_var.append(y_pred_test_pred_var_artifical)

                    X_train_val_manip = pd.concat([X_train_val_manip, X_test_manip.iloc[[0]]])
                    X_test_manip = X_test_manip.iloc[1:]
                    if hasattr(model, 'sequential'):
                        x_test = x_test[1:]
                    if (i+1) % period == 0:
                        X_train_val_manip = X_train_val_manip[self.user_input_params["refit_drops"]:]
                        if model.pca_transform:
                            X_train_val_manip, X_test_manip = \
                                self.pca_transform_train_test(X_train_val_manip, X_test_manip)
                        model.update(update=X_train_val_manip, period=period)
                        if hasattr(model, 'sequential'):
                            x_test = model.X_scaler.transform(X_test_manip.drop(labels=[self.target_column], axis=1))
                            y_test = model.y_scaler.transform(X_test_manip[self.target_column].values.reshape(-1, 1))
                            x_test, _ = model.create_sequences(x_test, y_test)

                y_pred_test = np.array(y_pred_test).flatten()
                y_pred_test_var = np.array(y_pred_test_var).flatten()
                if hasattr(final_model, 'conf'):
                    if np.array(y_pred_test_conf).ndim == 2:
                        y_pred_test_conf = np.reshape(np.array(y_pred_test_conf), (-1, 2))
                    else:
                        y_pred_test_conf = np.array(y_pred_test_conf).flatten()

            no_trials = len(self.study.trials) - 1 \
                if (self.user_input_params["intermediate_results_interval"] is not None) and \
                   (len(self.study.trials) % self.user_input_params["intermediate_results_interval"] != 0) \
                else len(self.study.trials)
            self.write_runtime_csv(dict_runtime={'Trial': 'retraining_after_' + str(no_trials) + '_trials',
                                                 'refitting_cycle': period,
                                                 'process_time_s': time.process_time() - start_process_time,
                                                 'real_time_s': time.time() - start_realclock_time,
                                                 'params': self.study.best_trial_copy.params, 'note': 'successful'})

            # Evaluate and save results
            if 'lstm' in self.user_input_params['current_model_name']:
                test = self.dataset.tail(test_len)
            eval_scores = eval_metrics.get_evaluation_report(y_true=test[self.target_column], y_pred=y_pred_test,
                                                             prefix='test_refitting_period_' + str(period) + '_',
                                                             current_model_name=self.user_input_params['current_model_name'])

            if self.user_input_params['current_model_name'] in ['ard', 'bayesridge', 'elasticnet', 'lasso', 'ridge', 'xgboost']:
                feat_import_df = self.get_feature_importance(model=model, period=period)
                feature_importance = pd.concat([feature_importance, feat_import_df], axis=1)

            print('## Results on test set with refitting period: ' + str(period) + ' ##')
            print(eval_scores)
            final_results.at[0:len(y_pred_test) - 1, 'y_pred_test_refitting_period_' + str(period)] = y_pred_test
            final_results.at[0:len(y_pred_test_var) - 1, 'y_pred_test_var_refitting_period_' + str(period)] = \
                y_pred_test_var
            if hasattr(final_model, 'conf'):
                if y_pred_test_conf.ndim == 2:
                    final_results.at[0:len(y_pred_test_conf) - 1, 'y_pred_test_lower_bound_refitting_period_' +
                                                                  str(period)] = y_pred_test_conf[:, 0].flatten()
                    final_results.at[0:len(y_pred_test_conf) - 1, 'y_pred_test_upper_bound_refitting_period_' +
                                                                  str(period)] = y_pred_test_conf[:, 1].flatten()
                else:
                    final_results.at[0:len(y_pred_test_conf) - 1, 'y_pred_test_conf_refitting_period_' + str(period)] \
                        = y_pred_test_conf.flatten()

            for metric, value in eval_scores.items():
                final_results.at[0, metric] = value
            if count == 0:
                final_eval_scores = eval_scores
            else:
                final_eval_scores = {**final_eval_scores, **eval_scores}

        if len(self.study.trials) == self.user_input_params["n_trials"]:
            results_filename = 'final_model_test_results.csv'
            feat_import_filename = 'final_model_feature_importances.csv'
            if self.user_input_params["save_final_model"]:
                final_model.save_model(path=self.save_path, filename='final_retrained_model')
        else:
            results_filename = 'intermediate_after_' + str(len(self.study.trials) - 1) + '_test_results.csv'
            feat_import_filename = \
                'intermediate_after_' + str(len(self.study.trials) - 1) + '_feat_importances.csv'
            shutil.copyfile(self.save_path.joinpath(self.current_model_name + '_runtime_overview.csv'),
                            self.save_path.joinpath('temp',
                                                    'intermediate_after_' + str(len(self.study.trials) - 1) + '_' +
                                                    self.current_model_name + '_runtime_overview.csv'), )
        final_results.to_csv(
            self.save_path.joinpath(postfix, results_filename),
            sep=',', decimal='.', float_format='%.10f', index=False
        )
        if self.user_input_params['current_model_name'] in ['ard', 'bayesridge', 'elasticnet', 'lasso', 'ridge', 'xgboost']:
            feature_importance.to_csv(
                self.save_path.joinpath(postfix, feat_import_filename),
                sep=',', decimal='.', float_format='%.10f', index=False
            )

        self.plot_results(final_results)

        return final_eval_scores

    def get_feature_importance(self, model: _base_model.BaseModel, period: int) -> pd.DataFrame:
        """
        Get feature importances for models that possess such a feature, e.g. XGBoost

        :param model: model to analyze
        :param period: refitting period

        :return: DataFrame with feature importance information
        """
        feat_import_df = pd.DataFrame()
        if self.user_input_params['current_model_name'] in ['xgboost']:
            feature_importances = model.model.feature_importances_
            sorted_idx = feature_importances.argsort()[::-1]
            feat_import_df['feature_period_' + str(period)] = \
                self.featureset.drop(self.target_column, axis=1).columns[sorted_idx]
            feat_import_df['feature_importance'] = feature_importances[sorted_idx]
        else:
            coef = model.model.coef_.flatten()
            sorted_idx = coef.argsort()[::-1]
            feat_import_df['feature_period_' + str(period)] = self.featureset.drop(self.target_column, axis=1).columns[sorted_idx]
            feat_import_df['coefficients'] = coef[sorted_idx]

        return feat_import_df

    def plot_results(self, final_results: pd.DataFrame):
        best_rmse = 999999999
        for periodical_refit_cycle in self.user_input_params['periodical_refit_frequency']:
            if final_results['test_refitting_period_' + str(periodical_refit_cycle) + '_rmse'].iloc[0] < best_rmse:
                best_refitting_cycle = periodical_refit_cycle
                best_rmse = final_results['test_refitting_period_' + str(periodical_refit_cycle) + '_rmse'].iloc[0]
        RMSE = round(best_rmse)

        pred = final_results['y_pred_test_refitting_period_' + str(best_refitting_cycle)].dropna().values
        true = final_results['y_true_test'].dropna().values

        x = list(range(1, len(pred) + 1))

        ax = plt.axes((0.1, 0.1, 1.0, 0.8))
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.plot(x, pred, marker='o', markersize=4, markerfacecolor='#c1272d',
                markeredgecolor='black', markeredgewidth=1, color='#c1272d',
                linestyle='--', linewidth=2)
        ax.plot(x, true, marker='o', markersize=4, markerfacecolor='#0000a7',
                markeredgecolor='black', markeredgewidth=1, color='#0000a7',
                linestyle='--', linewidth=2)

        ax.grid(True, which='major', axis='both', alpha=0.3)

        plt.xlabel("Time Step")
        plt.ylabel(self.target_column)

        if len(x) > 25:
            ax.xaxis.set_major_locator(plt.MaxNLocator(25))
        else:
            plt.xticks(x)

        ax.legend(['Forecasts(' + str(best_refitting_cycle) + ') with ' + self.user_input_params['current_model_name'] + ' (RMSE: ' + str(RMSE) + ')', "True Data"], frameon=False)

        fig1 = plt.gcf()
        plt.show()
        plt.draw()

        fig1.savefig(self.save_path.joinpath(self.user_input_params['current_model_name'] + '_' +
                                            self.user_input_params['featureset_name'] + '_' + 'best_refitting_cycle' +
                                            '_' + str(best_refitting_cycle) + '.png'), format='png', bbox_inches='tight')

    @property
    def run_optuna_optimization(self) -> dict:
        """
        Run whole optuna optimization for one model, dataset and datasplit.

        :return: dictionary with results overview
        """
        helper_functions.set_all_seeds()
        overall_results = {}
        print("## Starting Optimization")
        self.save_path = self.base_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # Create a new study
        self.study = self.create_new_study()
        self.current_best_val_result = None
        # Start optimization run
        self.study.optimize(lambda trial: self.objective(trial=trial), n_trials=self.user_input_params["n_trials"])
        self.study.best_trial_copy = self.study.best_trial
        helper_functions.set_all_seeds()
        # Calculate runtime metrics after finishing optimization
        runtime_metrics = self.calc_runtime_stats()
        # Print statistics after run
        print("## Optuna Study finished ##")
        print("Study statistics: ")
        print("  Finished trials: ", len(self.study.trials))
        print("  Pruned trials: ", len(self.study.get_trials(states=(optuna.trial.TrialState.PRUNED,))))
        print("  Completed trials: ", len(self.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))))
        print("  Best Trial: ", self.study.best_trial_copy.number)
        print("  Value: ", self.study.best_trial_copy.value)
        print("  Params: ")
        for key, value in self.study.best_trial_copy.params.items():
            print("    {}: {}".format(key, value))

        # Move validation results and models of best trial
        # files_to_keep = glob.glob(self.save_path + 'temp/' + '*trial' + str(self.study.best_trial_copy.number) + '*')
        files_to_keep_path = self.save_path.joinpath('temp', '*trial' + str(self.study.best_trial_copy.number) + '*')
        files_to_keep = pathlib.Path(files_to_keep_path.parent).expanduser().glob(files_to_keep_path.name)
        for file in files_to_keep:
            shutil.copyfile(file, self.save_path.joinpath(file.name))

        # Retrain on full train + val data with best hyperparams and apply on test
        for retry in range(len(self.best_trials)):
            try:
                final_eval_scores = self.generate_results_on_test()
            except ValueError as exc:
                print(traceback.format_exc())
                print(exc)
                self.study.best_trial_copy = \
                    [trial for trial in self.study.trials if trial.number == self.best_trials[retry + 1]][0]
                print('Testing failed. Will try again with next best model. The statistics of this study are:')
                print("  Finished trials: ", len(self.study.trials))
                print("  Pruned trials: ", len(self.study.get_trials(states=(optuna.trial.TrialState.PRUNED,))))
                print("  Completed trials: ", len(self.study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))))
                print("  Best Trial: ", self.study.best_trial_copy.number)
                print("  Value: ", self.study.best_trial_copy.value)
                print("  Params: ")
                for key, value in self.study.best_trial_copy.params.items():
                    print("    {}: {}".format(key, value))

                # files_to_keep = glob.glob(self.save_path + 'temp/' + '*trial' + str(self.study.best_trial_copy.number) + '*')
                files_to_keep_path = self.save_path.joinpath('temp', '*trial' + str(self.study.best_trial_copy.number) + '*')
                files_to_keep = pathlib.Path(files_to_keep_path.parent).expanduser().glob(files_to_keep_path.name)
                for file in files_to_keep:
                    shutil.copyfile(file, self.save_path.joinpath(file.name))
                continue
            shutil.rmtree(self.save_path.joinpath('temp'))
            break
        overall_results['Test'] = {'best_params': self.study.best_trial_copy.params, 'eval_metrics': final_eval_scores,
                                   'runtime_metrics': runtime_metrics, 'retries': retry}

        return overall_results
