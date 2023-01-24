Data Guide
===================
To run ForeTiS on your data, you need to provide a CSV or HDF5/H5/H5PY file like described below.
ForeTiS is designed to work with several file types besides the one we provide in this repository as tutorial data.

HDF5 / H5 / H5PY
~~~~~~~~~~~~~~~~~~~
The file can contain the following keys:

- **dataset_weather:** dataset containing only weather information
- **dataset_cal:** dataset containing only calendar information
- **dataset_sales:** dataset containing only sales information
- **dataset_sales_corr:** dataset containing only sales information with products correlating with the focus product
- **dataset_weather_sales:** dataset containing only weather and sales information
- **dataset_weather_sales_corr:** dataset containing only weather and sales information with products correlating with the focus product
- **dataset_weather_cal:** dataset containing only weather and calendar information
- **dataset_cal_sales:** dataset containing only calendar and sales information
- **dataset_cal_sales_corr:** dataset containing only calendar and sales information with products correlating with the focus product
- **dataset_full:** dataset containing weather, calendar and sales information
- **dataset_full_corr:** dataset containing weather, calendar and sales information with products correlating with the focus product

Depending on the datasets contains in the HDF5 file, you have to adjust the hyperparameters at
ForeTiS.model._base_model.BaseModel.dataset_hyperparam.

CSV
~~~~~
To use your own CSV data, the dataset must be in such a manner that the dataset_specific_config.ini file can be filled
like described below.

dataset_specific_config.ini
----------------------------
In this file you can define some characteristics of your data. The following points should be adjusted:

- **resolution:** if the resolution of the dataset is daily or weekly
- **resample_weekly:** whether the data should be resampled daily
- **seasonal_periods:** the length of one season of the data (e.g. for weekly data 52)
- **datatype:** if the data is in german (decimal=',', seperator=';') or american datatype (decimal='.', seperator=',')
- **date_column:** the name of the column that contains the date
- **holiday_school_column:** the name of the column that contains the school holidays
- **holiday_public_column:** the name of the column that contains the public holidays
- **special_days:** days of the year that are very important for your prediction task (e.g. Valentine's Day)
- **features_sales_regex:** the name of the columns that contain sales information (also possible as regex)
- **features_weather_regex:** the name of the columns that contain weather information (also possible as regex)
- **imputation:** whether imputation should be performed
- **cols_to_condense:** here you can define some names of columns that should be condensed
- **condensed_col_name:** the name of the new column created from the condensed columns

Preprocessing
----------------
In the preprocessing step, the actions defined in the dataset_specific_config.ini file will be performed. Additionally,
useless columns get dropped and, if the amount of a focus product gets predicted, the correlating products gets calculated.
Afterwards, in the featureadding and resampling step, the feature engineering happens where additional useful statisical and
calendar features (like defined in the dataset_specific_config.ini file) get added and the categorical features get on hot encoded.
Then, if defined, the data gets resampled and the datasets like described in :ref:`HDF5 / H5 / H5PY` will be created and saved.
Once the dataset is preprocessed and the HDF5 file is generated, the algorithm recognized this when restarting experiments and
directly reads in the HDF5 file to avoid redoing the time consuming preprocessing step.

