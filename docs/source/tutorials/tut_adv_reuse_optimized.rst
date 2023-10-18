HowTo: Reuse optimized model
==================================================
ForeTiS enables the reuse of an optimized model:

- Run inference using final model on new data with or without retraining on the whole old data using best hyperparameter combination

We provide scripts to run these functions (prefix *run_*) with our :ref:`Docker workflow`, on which we will also focus
in this tutorial. If you want to use the functions directly (e.g. with the pip installed package),
please check the scripts and see which functions are called.

Run inference on new data
""""""""""""""""""""""""""""""""""""""""""
The main use case for this is that you get new samples and you want to apply a previously optimized model on them. The final model therefore has to be saved and can be retrained
using the found hyperparameters and old dataset. To apply this prediction model on new data, the structure of the old and new dataset need to match exactly!

To apply a final prediction model on new data, you have to run the following command:

    .. code-block::

        python3 -m easypheno.postprocess.run_inference -rd full_path_to_model_results -odd path_to_old_data -od name_of_old_dataset -ndd path_to_new_data -nd name_of_new_dataset -sd path_to_save_directory
        python3 -m ForeTiS.postprocess.run_inference -rd /Users/ge35tuv/Desktop/ForeTiS/docs/source/tutorials/tutorial_data/results/ard/2023-10-17_12-35-47_featureset-full_timeseries-cv_True_20_20 -odd /Users/ge35tuv/Desktop/ForeTiS/docs/source/tutorials/tutorial_data -od nike_sales_old -ndd /Users/ge35tuv/Desktop/ForeTiS/docs/source/tutorials/tutorial_data -nd nike_sales_new -sd docs/source/tutorials/tutorial_data

By doing so, a .csv file containing the predictions on the whole new dataset will be created by applying the final prediction model, eventually after retraining on the old dataset if the final model was not saved.