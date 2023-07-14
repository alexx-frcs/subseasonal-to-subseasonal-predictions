# subseasonal-to-subseasonal-predictions

Code for reproducing the results regarding the addition and the analysis of some features for sub-seasonal to seasonal temperature predictions. 

## Getting started 

### Data


Within the folder data, create two additional subfolders **data/dataframes** and **data/forecast/cfsv2_2011-2018**.
Download the SubseasonalRodeo dataset from _https://doi.org/10.7910/DVN/IHBANG_ and place it in **data/dataframes**.
Download the Reconstructed Precipitation and Temperature CFSv2 Forecasts for 2011-2018 from _https://doi.org/10.7910/DVN/CEFZLV_. Place the files _cfsv2_re-contest_tmp2m-56w.h5_, _cfsv2_re-contest_tmp2m-34w.h5_, _cfsv2_re-contest_prate-56w.h5_ and _cfsv2_re-contest_prate-34w.h5_ in **data/dataframes**. Place the other files in data/forecast/cfsv2_2011-2018.\

To create the data used in the notebooks, execute the Jupyter notebook **create_data_matrices2**, which calls the file .py **experiments_util2**. That will create an HDF file that contains the lagged data (you can choose the number of days) used for the predictions. Please set the target_horizon to "56w" and gt_id to "contest_tmp2m"

For the soil moisture data, download it from _https://zenodo.org/record/7072512#.ZFj3h-zMK3J_, shrink it to 27;49 latitudes and -124;-94 longitudes and add it in the folder **data/new features**. \

For CSIF, go on _https://osf.io/8xqy6/_ and download every files with this format _OCO2.SIF.clear.inst.2000001.v2.nc_ and add it in **data/new features**. \

For El Nino, just execute the code related to this feature. 

### Methods

The main analysis have been conducted on soil moisture and elevation features addition. You can execute the Jupyter Notebook **cleaned xgboost CASM and elevation**, which calls the file .py **xgboost functional**.  \

In this notebook, you can visualize the seasonal shap values for all the features used in the model with the function **shap_geographical_analysis**. For the other functions, they're mainly focused on soil moisture and elevation. They can be easily adapted to other features later.\ 

For example, you can find the geographical skills improvements with CSIF addition in the Jupyter notebook **CSIF analysis**. In the Jupyter Notebook **tentative xgboost**, you have the model with El Nino feature.\

As the predictions and the shap values can take a long time to compute, you can store them and recover the files with the functions starting with 'recover'.
