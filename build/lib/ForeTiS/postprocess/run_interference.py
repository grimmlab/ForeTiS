import argparse
import pathlib
from pathlib import Path

from . import model_reuse

if __name__ == "__main__":
    """
    Run to apply the specified model on a dataset containing new samples.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-odd", "--old_data_dir", type=str,
                        help="Provide the name of the old data directory")
    parser.add_argument("-od", "--old_data", type=str,
                        help="specify the old dataset that you used.")
    parser.add_argument("-ndd", "--new_data_dir", type=str,
                        help="Provide the name of the new data directory")
    parser.add_argument("-nd", "--new_data", type=str,
                        help="specify the new dataset that you want to use.")
    parser.add_argument("-sd", "--save_dir", type=str,
                        help="Define the save directory for the results.")
    parser.add_argument("-rd", "--results_dir_model", type=str,
                        help="Provide the full path of the directory where your results of the model "
                             "you want to use are stored")
    parser.add_argument("-con_path", "--config_file_path", type=str, default=None,
                        help="specify the path to the config file you want to use."
                             "Default: dataset_specific_config.ini in ForeTiS root folder")
    parser.add_argument("-rm", "--retrain_model", type=bool, default=True,
                        help="specify whether to retrain the model with the whole old dataset."
                             "Standard is True")
    args = vars(parser.parse_args())
    new_data_dir = args['new_data_dir']
    new_data = args['new_data']
    old_data_dir = args['old_data_dir']
    old_data = args['old_data']
    save_dir = args["save_dir"]
    results_directory_model = args['results_dir_model']
    retrain_model = args['retrain_model']
    config_file_path = args['config_file_path']
    config_file_path = pathlib.Path(config_file_path) if config_file_path is not None else \
        Path(__file__).parent.parent.joinpath('dataset_specific_config.ini')

    model_reuse.apply_final_model(
        results_directory_model=results_directory_model, old_data_dir=old_data_dir, old_data=old_data,
        new_data_dir=new_data_dir, new_data=new_data, save_dir=save_dir, config_file_path=config_file_path,
        retrain_model=retrain_model
    )
