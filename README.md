# subseasonal-to-subseasonal-predictions

Code for reproducing the results regarding the addition and the analysis of some features for sub-seasonal to seasonal temperature predictions. 

## Getting started 

### Data

To create the data used in the notebooks, execute the Jupyter notebook **create_data_matrices2**, which calls the file .py **experiments_util2**. That will create an HDF file that contains the lagged data (you can choose the number of days) used for the predictions.

### Methods

The main analysis have been conducted on soil moisture and elevation features addition. You can execute the Jupyter Notebook **cleaned xgboost CASM and elevation**, which calls the file .py **xgboost functional**.   
