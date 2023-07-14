import pandas as pd
from datetime import datetime, timedelta
import xgboost as xgb
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import fsspec
import gcsfs
import os
import shap
import tables
import dask.dataframe as dd
import xarray as xr
import numpy as np
import requests
import time
import cartopy.crs as ccrs
import statistics
import cartopy
import seaborn as sns
import matplotlib.pyplot as plt

from skill import *
from experiments_util import *


gt_id='contest_tmp2m' #'contest_tmp2m' or 'contest_precip'
target_horizon='56w' #'56w' or '34w'


def import_data():
    t=time.time()
    test=pd.read_hdf('results/regression1/shared/contest_tmp2m_56w/lat_lon_date_data-contest_tmp2m_56w.h5',key='data')
    print('Elapsed time=',time.time()-t)
    return test

def add_casm(data):
    t=time.time()
    df_CASM=pd.DataFrame()
    df_CASM['lat']=data['lat']
    df_CASM['lon']=data['lon']
    df_CASM['CASM']=data['CASM_soil_moisture_shift44']
    df_CASM['start_date']=data['start_date']

    df_CASM_san=df_CASM.dropna(subset=['CASM'])

    df_CASM_san['lat'] = df_CASM_san['lat'].fillna(0)
    df_CASM_san['lon'] = df_CASM_san['lon'].fillna(0)


    df_CASM_san['lat']=df_CASM_san['lat'].round()
    df_CASM_san['lon']=df_CASM_san['lon'].round()

    new_df = df_CASM_san.groupby(['lat', 'lon', 'start_date'], as_index=False)['CASM'].mean()

    merged_df = data.merge(new_df[['lat', 'lon', 'start_date', 'CASM']], on=['lat', 'lon', 'start_date'], how='left')
    merged_df = merged_df.rename(columns={'CASM': 'CASM_new'})
    data['CASM'] = merged_df['CASM_new']

    df_filtered = data.dropna(subset=['tmp2m_anom'])
    print('CASM Loading time=',time.time()-t)
    return df_filtered

def add_elevation(data):
    t=time.time()
    dataset = xr.open_dataset('data/new features/elevationdata.nc')
    latitudes = dataset['lat'].values


    latitudes=np.repeat(latitudes,1386)
    longitudes = dataset['lon'].values
    elevations=dataset['elevation'].values


    gt = pd.DataFrame({
        'lat': latitudes
    })

    longitudes_repeated = np.tile(longitudes, 585)
    df_lon = pd.DataFrame({'lon': longitudes_repeated})
    elev=[]
    for lat in elevations:
        for lon in lat:
            elev.append(lon)
    df_elev=pd.DataFrame({'elevation':elev})
    gt=pd.concat([gt,df_lon,df_elev],axis=1)
    if isinstance(gt.index, pd.MultiIndex):
        gt.reset_index(inplace=True)

    lat_restriction_left = 27 #restriction of data around the US (latitude North)
    lat_restriction_right = 49 #restriction of data around the US (latitude North)
    lon_restriction_left = -124 #restriction of data around the US (longitude West)
    lon_restriction_right = -94 #restriction of data aroud the US (longitude West)

    #only keep the values inside
    gt = gt[gt['lat'].between(lat_restriction_left, lat_restriction_right)] 
    gt = gt[gt['lon'].between(lon_restriction_left, lon_restriction_right)]

    gt['lon'] = np.where(gt['lon']< 0, gt['lon'] + 360, gt['lon'])

    gt['lat_rounded']=gt['lat'].round()
    gt['lon_rounded']=gt['lon'].round()

    gt_new = gt.groupby(['lat_rounded', 'lon_rounded'], as_index=False)['elevation'].mean()

    gt_new=gt_new.rename(columns={'lat_rounded':'lat','lon_rounded':'lon'})
    merged_df = pd.merge(data, gt_new, on=['lat', 'lon'], how='left')

    print('Elevation loading time=',time.time()-t)
    return merged_df
    
    
def xgboost_regressor_initial_training_for_data_analysis(df,gt_id,target_horizon,year,features):
    """
    target_date: datetime object, date to predict
    """
    #choose the features you want to include
    print(features)
    train_data = df[df['start_date'].dt.year != year]
    test_data = df[df['start_date'].dt.year == year]
    
    X_train=train_data[[col for col in features]]
    y_train=train_data['tmp2m_anom']
    
    X_test = test_data[[col for col in features]]
    y_test_actual = test_data['tmp2m_anom']
    
    model=xgb.XGBRegressor(n_estimators=100)
    model.fit(X_train,y_train)
    
    y_test_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test_actual, y_test_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_test_pred)
    results = pd.DataFrame({'Date': test_data['start_date'], 'Actual Temperature': y_test_actual, 'Predicted Temperature': y_test_pred,
                         'RMSE':rmse,'MAE':mae})
    return y_test_actual, y_test_pred, results

def compute_predictions(df,gt_id,target_horizon):
    
    features_casm=['tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86', 'tmp2m_shift86_anom',
                                'rhum_shift44', 'pres_shift44',
                                'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa','CASM']
    features_notcasm=['tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86', 'tmp2m_shift86_anom',
                            'rhum_shift44', 'pres_shift44',
                            'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa']
    features_casm_elevation=['tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86', 'tmp2m_shift86_anom',
                            'rhum_shift44', 'pres_shift44',
                            'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa','CASM','elevation']
    years=[2011+i for i in range(7)]
    predictions_with_casm = {            2011: { 
            'results':pd.DataFrame()
                  },            
        2012: { 
                           'results':pd.DataFrame()

                  },
            2013: { 
                               'results':pd.DataFrame()

                  },
        2014: {
                           'results':pd.DataFrame()

        },
        2015: {
            
                        'results':pd.DataFrame()

        },
        2016: {
           
                        'results':pd.DataFrame()

        },
        2017: { 
                           'results':pd.DataFrame()

          },
    }
    for year in years:
        t=time.time()
        y_test_actual, y_test_pred, results= xgboost_regressor_initial_training_for_data_analysis(df,gt_id,target_horizon,year,features_casm)
        predictions_with_casm[year]['results']=results
        print('{}'.format(year)+' is done')
        print('Elapsed time for {}={}'.format(year,time.time()-t))
        
       #then without casm 
    predictions_without_casm = {            2011: { 
                                                               'results':pd.DataFrame()

                  },            
        2012: { 
                           'results':pd.DataFrame()

                  },
            2013: { 
                               'results':pd.DataFrame()

                  },
        2014: { 
                           'results':pd.DataFrame()

        },
        2015: {
                        'results':pd.DataFrame()

        },
        2016: {
                        'results':pd.DataFrame()

        },
        2017: {
                           'results':pd.DataFrame()

          }
    }
    for year in years:
        t=time.time()
        y_test_actual, y_test_pred, results= xgboost_regressor_initial_training_for_data_analysis(df,gt_id,target_horizon,year,features_notcasm)
        predictions_without_casm[year]['results']=results
        print('{}'.format(year)+' is done')
        print('Elapsed time for {}={}'.format(year,time.time()-t))
    predictions_with_casm_and_elevation = {            2011: { 
                                                               'results':pd.DataFrame()

                  },            
        2012: { 
                           'results':pd.DataFrame()

                  },
            2013: { 
                               'results':pd.DataFrame()

                  },
        2014: { 
                           'results':pd.DataFrame()

        },
        2015: {
                        'results':pd.DataFrame()

        },
        2016: {
                        'results':pd.DataFrame()

        },
        2017: {
                           'results':pd.DataFrame()

          }
    }
    for year in years:
        t=time.time()
        y_test_actual, y_test_pred, results= xgboost_regressor_initial_training_for_data_analysis(df,gt_id,target_horizon,year,features_casm_elevation)
        predictions_with_casm_and_elevation[year]['results']=results
        print('{}'.format(year)+' is done')
        print('Elapsed time for {}={}'.format(year,time.time()-t))
    return predictions_with_casm_and_elevation,predictions_with_casm,predictions_without_casm

def recover_predictions():
    predictions_with_casm_and_elevation = {            2011: { 
                                                           'results':pd.DataFrame()

              },            
    2012: { 
                       'results':pd.DataFrame()

              },
        2013: { 
                           'results':pd.DataFrame()

              },
    2014: { 
                       'results':pd.DataFrame()

    },
    2015: {
                    'results':pd.DataFrame()

    },
    2016: {
                    'results':pd.DataFrame()

    },
    2017: {
                       'results':pd.DataFrame()

      }
    }
    predictions_with_casm = {2011: { 
                                                       'results':pd.DataFrame()

          },            
    2012: { 
                   'results':pd.DataFrame()

          },
    2013: { 
                       'results':pd.DataFrame()

          },
    2014: { 
                   'results':pd.DataFrame()

    },
    2015: {
                'results':pd.DataFrame()

    },
    2016: {
                'results':pd.DataFrame()

    },
    2017: {
                   'results':pd.DataFrame()

      }
    }
    predictions_without_casm = {            2011: { 
                                                       'results':pd.DataFrame()

          },            
    2012: { 
                   'results':pd.DataFrame()

          },
    2013: { 
                       'results':pd.DataFrame()

          },
    2014: { 
                   'results':pd.DataFrame()

    },
    2015: {
                'results':pd.DataFrame()

    },
    2016: {
                'results':pd.DataFrame()

    },
    2017: {
                   'results':pd.DataFrame()

      }
    }

    for year in range(2011,2018):
        predictions_with_casm_and_elevation[year]['results']=pd.read_hdf('results/predictions/rodeo_casm_elevation.h5',key='/{}/results'.format(year))
    for year in range(2011,2018):
        predictions_with_casm[year]['results']=pd.read_hdf('results/predictions/rodeo_casm.h5',key='/{}/results'.format(year))
    for year in range(2011,2018):
        predictions_without_casm[year]['results']=pd.read_hdf('results/predictions/rodeo_only.h5',key='/{}/results'.format(year))
    return predictions_with_casm_and_elevation,predictions_with_casm,predictions_without_casm

def plot_predictions(predictions_without_casm,predictions_with_casm,predictions_with_casm_and_elevation,point,year,feature_string):
    if feature_string=='rodeo':
        plot_df=predictions_without_casm[year]['results'].copy()
    elif feature_string=='casm':
        plot_df=predictions_with_casm[year]['results'].copy()
    elif feature_string=='casm and elevation':
        plot_df=predictions_with_casm_and_elevation[year]['results'].copy()
    plot_df.reset_index(inplace=True,drop=True)

    start_point = (point-1) * 365
    end_point = point * 365

    plot_df = plot_df.iloc[start_point:end_point]
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(plot_df['Date'], plot_df['Actual Temperature'], label='Ground Truth', color='blue')
    ax1.set_ylabel('Temperature (Ground Truth)', color='blue')

    ax2.plot(plot_df['Date'], plot_df['Predicted Temperature'], label='Predictions with {}'.format(feature_string), color='red')
    ax2.set_ylabel('Temperature (Predictions)', color='red')

    lines = [ax1.get_lines()[0], ax2.get_lines()[0]]
    ax1.legend(lines, [line.get_label() for line in lines])

    ax1.set_xlabel('Dates')

    plt.show()

    


def recover_shap_values():
    shap_values_good= {
    2011:[],
    2012:[],
    2013:[],
    2014:[],
    2015:[],
    2016:[],
    2017:[],
    }
    hdf_file = 'results/shap_values/casm_elevation_good.h5'
    year=2011
    with h5py.File(hdf_file, 'r') as file:
        for dataset_name, dataset in file.items():
            data = dataset[:]
            shap_values_good[year]=data
            year+=1
    return shap_values_good
            



def show_skills(plot,predictions_with_casm_and_elevation,predictions_with_casm,predictions_without_casm):
    skills_with_casm_elevation=[]
    skills_with_casm=[]
    skills_without_casm=[]
    df_cfsv2=pd.read_hdf('results/skills/debiased_cfsv2/skill-contest_tmp2m-56w.h5')
    df_cfsv2.set_index('start_date',inplace=True)
    skills_cfsv2 = df_cfsv2.resample('Y').mean()
    skills_cfsv2['year'] = skills_cfsv2.index.year
    
    result_with_casm=pd.DataFrame()
    result_without_casm=pd.DataFrame()
    result_with_casm_elevation=pd.DataFrame()
    
    for year in range(2011,2018):
        skill_without_casm=get_col_skill(predictions_without_casm[year]['results'],'Actual Temperature','Predicted Temperature'
                                        ,date_col='Date')
        skills_without_casm.append(skill_without_casm)
        skill_with_casm=get_col_skill(predictions_with_casm[year]['results'],'Actual Temperature','Predicted Temperature'
                                      ,date_col='Date')
        skills_with_casm.append(skill_with_casm)
        skill_with_casm_elevation=get_col_skill(predictions_with_casm_and_elevation[year]['results'],'Actual Temperature','Predicted Temperature'
                                        ,date_col='Date')
        skills_with_casm_elevation.append(skill_with_casm_elevation)
        
        
    result_with_casm_elevation['skill']=skills_with_casm_elevation
    result_with_casm_elevation['year']=[i for i in range(2011,2018)]
    result_with_casm['skill']=skills_with_casm
    result_with_casm['year']=[i for i in range(2011,2018)]
    result_without_casm['skill']=skills_without_casm
    result_without_casm['year']=[i for i in range(2011,2018)]
    print(result_with_casm,result_without_casm)
    if plot=='Rodeo XGBoost alone':
        plt.plot(result_without_casm['year'],result_without_casm['skill'],label=plot)
        plt.legend()
        plt.xlabel('Year')
        plt.ylabel('Skill')

        plt.show()
    elif plot=='Rodeo XGBoost vs CASM XGBoost':
        plt.plot(result_without_casm['year'],result_without_casm['skill'],label='XGBoost with Rodeo only')
        plt.plot(result_with_casm['year'],result_with_casm['skill'],label='XGBoost with Rodeo and CASM')
        for i in range(len(result_with_casm['year'])):
            plt.plot([result_without_casm['year'][i], result_with_casm['year'][i]], [result_without_casm['skill'][i], result_with_casm['skill'][i]], 'k--')
            diff_percentage = (result_with_casm['skill'][i] - result_without_casm['skill'][i]) / result_without_casm['skill'][i] * 100 
            plt.text((result_without_casm['year'][i] + result_with_casm['year'][i]) / 2, (result_without_casm['skill'][i] + result_with_casm['skill'][i]) / 2+0.013, f'{diff_percentage:.2f}%', ha='center', va='center')  
            plt.text(2014.3,0.1,'Improvement mean=4%')

        plt.legend()
        plt.xlabel('Year')
        plt.ylabel('Skill')

        plt.show()
    elif plot=='Rodeo XGBoost vs CFSv2':
        plt.plot(skills_cfsv2['year'],skills_cfsv2['skill'],label='debiased CFSv2')
        plt.plot(result_without_casm['year'],result_without_casm['skill'],label='Rodeo XGBoost')
        plt.legend()
        plt.xlabel('Year')
        plt.ylabel('Skill')

        plt.show()
    elif plot=='CASM XGBoost vs CASM and Elevation XGBoost':
        plt.plot(result_with_casm_elevation['year'],result_with_casm_elevation['skill'],label='CASM and elevation XGBoost')
        plt.plot(result_with_casm['year'],result_with_casm['skill'],label='CASM XGBoost')
        plt.legend()
        plt.xlabel('Year')
        plt.ylabel('Skill')

        plt.show()
        
def calculate_shap_values_for_data_analysis(df,gt_id,target_horizon,year,features):

    train_data = df[df['start_date'].dt.year != year]
    test_data = df[df['start_date'].dt.year == year]

    X_train=train_data[[col for col in features]]
    y_train=train_data['tmp2m_anom']

    X_test = test_data[[col for col in features]]
    y_test_actual = test_data['tmp2m_anom']


    model=xgb.XGBRegressor(n_estimators=300)
    model.fit(X_train,y_train)

    explainer = shap.Explainer(model)

    # Calculer les valeurs de SHAP
    shap_values = explainer.shap_values(X_test)

    return shap_values

def compute_shap_values(df,gt_id,target_horizon,features):
    t=time.time()
    shap_values = {
    2011:[],
    2012:[],
    2013:[],
    2014:[],
    2015:[],
    2016:[],
    2017:[],
    }
    for i in range (7):
        t=time.time()
        shap_values[2011+i]=calculate_shap_values_for_data_analysis(df,gt_id,target_horizon,2011+i,features)
        print('Elapsed time for {}='.format(2011+i),time.time()-t)
    hdf_directory = 'results/shap_values'

    hdf_filename = 'casm_elevation.h5'

    hdf_filepath = os.path.join(hdf_directory, hdf_filename)
    
    with h5py.File(hdf_filepath, 'w') as hf:
        for year, array in shap_values.items():
            hf.create_dataset(str(year), data=array)
    dataframe_indices=df[df['start_date'].dt.year>=2011][['start_date','lat','lon','tmp2m','tmp2m_anom']]
    #resort the dataframe, so the values are sorted regarding the date, then the latitude and then the longitude

    # create 'index' column, so every (lat,lon) couple has a unique encoding number
    dataframe_indices.reset_index(drop=True, inplace=True)
    dataframe_indices['index'] = dataframe_indices.groupby(['lat', 'lon']).ngroup()
    indexes=dataframe_indices['index'].values
    dataframe_indices['Global index']=[i for i in range(1386772)]
    dataframe_indices['season'] = dataframe_indices['start_date'].apply(assign_season)
    dataframe_indices['day'] = dataframe_indices['start_date'].dt.dayofyear
    dataframe_indices = dataframe_indices.sort_values(by=['lat','lon','start_date'])
    
    hdf_filename = 'dataframe_indices.h5'
    hdf_filepath = os.path.join(hdf_directory, hdf_filename)

    dataframe_indices.to_hdf(hdf_filepath, key='data', mode='w')


    
    shap_values_good= {
    2011:[],
    2012:[],
    2013:[],
    2014:[],
    2015:[],
    2016:[],
    2017:[],
    }
    #creation of a dataframe to keep track of indexes
    """
    pos=0
    for i in range (2011,2019):
        shap_values_with_CASM_and_elevation_good[i] = np.zeros((shap_values_with_CASM_and_elevation[i].shape[0], shap_values_with_CASM_and_elevation[i].shape[1]+1))  # Nouveau numpy array avec une colonne supplémentaire

        for j in range(shap_values_with_CASM_and_elevation[i].shape[0]):
            new_array = np.append(shap_values_with_CASM_and_elevation[i][j], pos)  # Ajouter la position dans le numpy array principal à la fin de chaque numpy array
            shap_values_with_CASM_and_elevation_good[i][j] = new_array
            pos+=1
    """
    print('Elapsed time=',time.time()-t)
    for i in range(2011,2018):
        t=time.time()
        shap_values_good[i] = np.zeros((shap_values[i].shape[0], shap_values[i].shape[1]+3))
        dataframe_year=dataframe_indices[dataframe_indices['start_date'].dt.year==i]
        for j in range(shap_values[i].shape[0]):
            new_array = np.append(shap_values[i][j],dataframe_year['index'].iloc[j])
            new_array=np.append(new_array,dataframe_year['season'].iloc[j])
            new_array=np.append(new_array,dataframe_year['day'].iloc[j])

            shap_values_good[i][j] = new_array

        print('{} is done'.format(i),'Elapsed time=',time.time()-t)
    return shap_values_good

        
        
def assign_season(date):
    year=date.year
    spring_start = pd.to_datetime('{}-03-21'.format(year))
    summer_start = pd.to_datetime('{}-06-22'.format(year))
    autumn_start = pd.to_datetime('{}-09-22'.format(year))
    winter_start = pd.to_datetime('{}-12-21'.format(year))
    if date >= spring_start and date < summer_start:
        return 1
    elif date >= summer_start and date < autumn_start:
        return 2
    elif date >= autumn_start and date < winter_start:
        return 3
    else:
        return 4
    
def plot_shap_summary(df,shap_values,year):
    candidate_x_cols = ['tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86', 'tmp2m_shift86_anom',
                            'rhum_shift44', 'pres_shift44',
                            'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa','GPP_shift44','RECO_shift44','CASM','elevation']
    train_data_new = df[df['start_date'].dt.year == 2015]
    X_train_new=train_data_new[[col for col in candidate_x_cols]]
    shap.summary_plot(shap_values[:,:12], X_train_new,plot_type='bar')
    
def plot_shap_scatter(df,shap_values,year,feature):
    df_function=df.copy()
    df_function['key_4']=df_function['key_2']
    df_shap=df_function[['start_date','tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86', 'tmp2m_shift86_anom',
                                'rhum_shift44', 'pres_shift44',
                                'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa','GPP_shift44','RECO_shift44','CASM','elevation','key_2','key_3','key_4'
                      ]]
    df_shap=df_shap[df_shap['start_date'].dt.year==year] #pick the good year
    df_shap.drop(columns=['start_date'],inplace=True,axis=1)

    shap_array=df_shap.values
    shap_exp = shap.Explanation(values=shap_values,
                            data=shap_array,
                            feature_names=['tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86', 'tmp2m_shift86_anom',
                                'rhum_shift44', 'pres_shift44',
                                'nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa','GPP_shift44','RECO_shift44','CASM','elevation','index','season','day'
                     ])
    shap.plots.scatter(shap_exp[:,feature])
    
def string_season(season):
    if season==1:
        return 'spring'
    elif season==2:
        return 'summer'
    elif season==3:
        return 'fall'
    else:
        return 'winter'

def shap_geographical_analysis(shap_array,feature_number,feature_string):
    
    df_shap_values=pd.DataFrame()
    index_coords=[i for i in range(514)]
    dataframe_indices=pd.read_hdf('results/shap_values/dataframe_indices.h5')


    fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(64, 32), subplot_kw={'projection': ccrs.LambertConformal()})
    for k,year in enumerate(range(2011,2018)):
        row=k
        season_values=shap_array[year][:, 13]
        for j in range(1,5):
            col = j-1
            shap_values=[]
            ax = axes[row,col]
            mask = np.logical_and(season_values==j, season_values == j)

            #this array only contains the shap values for the points in the relevant season
            numpy_arrays_good = shap_array[year][mask]

            index_values=numpy_arrays_good[:,12]
            for i in range(514):
                mask = np.logical_and(index_values==i, index_values == i)
                numpy_arrays_index=numpy_arrays_good[mask]
                shap_value=numpy_arrays_index[:,feature_number].mean()
                shap_values.append(shap_value)

            df_shap_values['Shap values']=shap_values
            df_shap_values['Index']=index_coords
            dataframe_indices_bis=dataframe_indices.sort_values(by=['start_date','lat','lon'])
            dataframe_indices_bis.reset_index(inplace=True,drop=True)
            df_shap_values['lat']=dataframe_indices_bis['lat'][0:514]
            df_shap_values['lon']=dataframe_indices_bis['lon'][0:514]

            # Définir la zone d'affichage de la carte sur les États-Unis
            ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())

            # Charger les limites des États-Unis avec cartopy
            ax.add_feature(cartopy.feature.STATES)

            marker_size=50

            # Tracer les données sur la carte en utilisant des couleurs basées sur la colonne 'skill'
            if feature_string=='CASM':
                vmin=0
                vmax=.08
            if feature_string=='elevation':
                vmin=-.08
                vmax=.1
            sc = ax.scatter(df_shap_values['lon'], df_shap_values['lat'],c=df_shap_values['Shap values'], marker='s', s=marker_size,cmap='RdYlGn', transform=ccrs.PlateCarree(),vmin=vmin,vmax=vmax)

            cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
            cbar.set_label('SHAP values for {}'.format(feature_string))


            # Ajouter un titre à la carte
            ax.set_title('{} mean shap values in \n{} {}'.format(feature_string,string_season(j),year))

    # Afficher la carte
    plt.tight_layout()

    plt.show()

    
def transform_large_region(row):
    if row['Large region']=='C':
        if row['lon'] < 240 or (row['lon'] >= 250 and row['lat'] < 31):
            return 'Co'
        else:
            return 'Cc'
    else:
        return row['Large region']

def plot_climate_regions():
    file_path = "data/clusters coordinates/koppen_1901-2010.tsv"

    df = pd.read_csv(file_path, delimiter='\t')
    df_climate_zones=df[(df['longitude']>=-125)&(df['longitude']<=-93)&(df['latitude']>=26)&(df['latitude']<=50)]
    df_climate_zones.reset_index(inplace=True,drop=True)
    df_climate_zones['longitude']=df_climate_zones['longitude']+360
    df_climate_zones = df_climate_zones.sort_values(by=['latitude', 'longitude'])
    df_climate_zones['latitude'] = df_climate_zones['latitude'].astype(float)
    df_climate_zones['longitude'] = df_climate_zones['longitude'].astype(float)
    df_climate_zones['Large region'] = df_climate_zones['p1901_2010'].apply(lambda x: x[0])
    df_climate_zones['latitude'] = df_climate_zones['latitude'].apply(lambda x: round(x) if round(x % 1, 2) == 0.25 else x)
    df_climate_zones['longitude'] = df_climate_zones['longitude'].apply(lambda x: round(x) if round(x % 1, 2) == 0.25 else x)
    
    df_climate_zones.rename(columns={'latitude':'lat','longitude':'lon'},inplace=True)
    df_intermediate=pd.read_hdf('results/shap_values/dataframe_indices.h5')
    df_coordinates = df_intermediate.drop_duplicates(['lat', 'lon'])[['lat','lon']]
    df_coordinates.reset_index(inplace=True,drop=True)
    df_coordinates=pd.merge(df_coordinates,df_climate_zones,on=['lat','lon'],how='left')
    df_coordinates['Large region'] = df_coordinates.apply(transform_large_region, axis=1)


    map = ccrs.LambertConformal(central_longitude=-95, central_latitude=37.5, standard_parallels=(33, 45))

    plt.figure(figsize=(12, 8))

    ax = plt.axes(projection=map)
    ax.set_extent([-130, -65, 23, 50])
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor='black')
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')

    zone_colors = {
        'B': 'red',
        'Cc':'blue',
        'Co': 'cyan',
        'D': 'green',
        'E': 'orange'
    }

    for index, row in df_coordinates.iterrows():
        lat = row['lat']
        lon = row['lon']
        climate_zone = row['Large region']

        ax.plot(lon, lat, marker='o', color=zone_colors.get(climate_zone, 'black'), markersize=5, transform=ccrs.PlateCarree())

    legend_labels = ['B - Dry climates', 'Cc - Mild temperate continental regions', 'Co - Mild temperate oceanic regions','D Continental', 'E - Polar']
    legend_colors = ['red', 'blue', 'cyan','green', 'orange']
    legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=label, linestyle='') for label, color in zip(legend_labels, legend_colors)]
    ax.legend(handles=legend_elements, loc='lower left')

    # Afficher la carte
    plt.show()
    return df_climate_zones

def plot_mean_shap(shap_array,df_values,df_climate_zones,feature_number):
    
    dataframe_indices=pd.read_hdf('results/shap_values/dataframe_indices.h5')
    feature='CASM'
    data_with_climate_zones=pd.merge(df_values,df_climate_zones,on=['lat','lon'],how='left')
    df_shap_values=pd.DataFrame()
    index_coords=[i for i in range(514)]


    # Define the grid layout for subplots
    fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(16, 20))

    # Legend labels for each column
    legend_labels = ['Arid', 'Mild cont', 'Mild Oc', 'Cont climate', 'Polar']

    # Iterate over years and seasons
    for k, year in enumerate(range(2011, 2018)):
        season_values = shap_array[year][:, 13]

        # Iterate over seasons
        for j in range(1, 5):
            shap_values = []
            mask = np.logical_and(season_values == j, season_values == j)
            numpy_arrays_good = shap_array[year][mask]
            index_values = numpy_arrays_good[:, 12]

            # Iterate over index values
            for i in range(514):
                mask = np.logical_and(index_values == i, index_values == i)
                numpy_arrays_index = numpy_arrays_good[mask]
                shap_value = numpy_arrays_index[:, feature_number].mean()
                shap_values.append(shap_value)

            dataframe_indices_bis = dataframe_indices.sort_values(by=['start_date', 'lat', 'lon'])
            dataframe_indices_bis.reset_index(inplace=True, drop=True)
            df_shap_values['lat'] = dataframe_indices_bis['lat'][0:514]
            df_shap_values['lon'] = dataframe_indices_bis['lon'][0:514]

            data_with_climate_zones_year = data_with_climate_zones[data_with_climate_zones['start_date'].dt.year == year][['lat', 'lon', 'Large region']]
            data_with_climate_zones_year = data_with_climate_zones_year.drop_duplicates(subset=['lat', 'lon', 'Large region'])
            data_with_climate_zones_year.reset_index(inplace=True, drop=True)
            df_shap_values['Climate region'] = data_with_climate_zones_year['Large region']
            df_shap_values['Shap values'] = shap_values
            df_shap_values['Index'] = index_coords

            # Calculate mean of 'Shap values' grouped by 'Climate region'
            grouped_df = df_shap_values.groupby('Climate region')['Shap values'].mean().reset_index()

            # Get the number of points for each 'Climate region'
            point_counts = df_shap_values['Climate region'].value_counts()
            point_counts = point_counts.reindex(grouped_df['Climate region'])
            grouped_df['Point counts'] = point_counts.values

            # Get the corresponding subplot axes
            row = k
            col = j - 1
            ax = axes[row, col]

            # Plot bar chart
            sns.barplot(x='Climate region', y='Shap values', data=grouped_df, ax=ax)
            ax.set_title('Year: {} - Season: {}'.format(year, string_season(j)))
            ax.set_xlabel('Climate region')
            ax.set_ylabel('Mean Shap values')

            # Annotate point counts on the bar chart
            for i, count in enumerate(point_counts.values):
                ax.text(i, grouped_df['Shap values'].iloc[i], str(count), ha='center', va='bottom')

            # Set legend labels for each column
            if col == 0:
                ax.legend(legend_labels, loc='upper left')

    # Remove empty subplots
    for k in range(7, 7):
        for j in range(4, 7):
            fig.delaxes(axes[k, j])

    # Adjust spacing between subplots
    fig.tight_layout()

    # Show the plot
    plt.show()
   

    
def seasonal_shap_importance(shap_values_with_CASM_and_elevation,boundary,treshold):
    casm_feature_importance={
        2011:{'spring':[],
              'summer':[],
              'fall':[],
              'winter':[]
        },
        2012:{'spring':[],
              'summer':[],
              'fall':[],
              'winter':[]
        },
        2013:{'spring':[],
              'summer':[],
              'fall':[],
              'winter':[]
        },
        2014:{'spring':[],
              'summer':[],
              'fall':[],
              'winter':[]
        },
        2015:{'spring':[],
              'summer':[],
              'fall':[],
              'winter':[]
        },
        2016:{'spring':[],
              'summer':[],
              'fall':[],
              'winter':[]
        },
        2017:{'spring':[],
              'summer':[],
              'fall':[],
              'winter':[]
        }
    }
    for year_to_study in range(2011,2018):
        for season_to_study in range(1,5):
            feature_shap_values=shap_values_with_CASM_and_elevation[year_to_study][:, 10]

            boundary=boundary
            if boundary>0:
                mask = np.logical_and(feature_shap_values<=boundary, feature_shap_values <= boundary)
            else:
                mask = np.logical_and(feature_shap_values>=boundary, feature_shap_values >= boundary)



            numpy_arrays_bad = shap_values_with_CASM_and_elevation[year_to_study][mask]


            numpy_arrays_good = shap_values_with_CASM_and_elevation[year_to_study][~mask]


            season_values=numpy_arrays_good[:,13]
            mask = np.logical_and(season_values== season_to_study , season_values== season_to_study)
            all_values =[arr[ 12] for arr in numpy_arrays_good[mask]]

            unique_values, value_counts = np.unique(all_values, return_counts=True)
            #unique_values contains all the points for which at least one date in the season to study has acknowledged more than "boundary" augmentation. 
            #value_counts contains, for every point in unique_values, the number of dates in the season to study that have acknowledged more than "boundary" augmentation.  
            treshold=treshold
            filtered_values_seasons = unique_values[value_counts >= treshold]
            #filtered_values_seasons contains all the points for which the number of dates in the season to study that have acknowledged more than "boundary" augmentation is above "treshold".
            if season_to_study==1:
                if  value_counts.size==0:
                    casm_feature_importance[year_to_study]['spring'].append(0)
                    casm_feature_importance[year_to_study]['spring'].append(len(filtered_values_seasons))

                else:
                    casm_feature_importance[year_to_study]['spring'].append(value_counts.mean())
                    casm_feature_importance[year_to_study]['spring'].append(len(filtered_values_seasons))
            if season_to_study==2:
                if  value_counts.size==0:
                    casm_feature_importance[year_to_study]['summer'].append(0)
                    casm_feature_importance[year_to_study]['summer'].append(len(filtered_values_seasons))

                else:
                    casm_feature_importance[year_to_study]['summer'].append(value_counts.mean())
                    casm_feature_importance[year_to_study]['summer'].append(len(filtered_values_seasons))
            if season_to_study==3:
                if  value_counts.size==0:
                    casm_feature_importance[year_to_study]['fall'].append(0)
                    casm_feature_importance[year_to_study]['fall'].append(len(filtered_values_seasons))

                else:
                    casm_feature_importance[year_to_study]['fall'].append(value_counts.mean())
                    casm_feature_importance[year_to_study]['fall'].append(len(filtered_values_seasons))
            if season_to_study==4:
                if  value_counts.size==0:
                    casm_feature_importance[year_to_study]['winter'].append(0)
                    casm_feature_importance[year_to_study]['winter'].append(len(filtered_values_seasons))                                                       
                else:
                    casm_feature_importance[year_to_study]['winter'].append(value_counts.mean())
                    casm_feature_importance[year_to_study]['winter'].append(len(filtered_values_seasons))  
                    
    years = list(casm_feature_importance.keys())
    spring_values = [casm_feature_importance[year]['spring'][0] for year in years]
    summer_values = [casm_feature_importance[year]['summer'][0] for year in years]
    fall_values = [casm_feature_importance[year]['fall'][0] for year in years]
    winter_values = [casm_feature_importance[year]['winter'][0] for year in years]
    if boundary>0:
        adverb='above'
    else:
        adverb='below'

    fig, ax = plt.subplots(figsize=(15,6))

    x = np.arange(len(years))
    bar_width = 0.15
    offset = np.array([-1.5, -0.5, 0.5, 1.5]) * bar_width

    ax.bar(x + offset[0], spring_values, width=bar_width, label='Spring')
    ax.bar(x + offset[1], summer_values, width=bar_width, label='Summer')
    ax.bar(x + offset[2], fall_values, width=bar_width, label='Fall')
    ax.bar(x + offset[3], winter_values, width=bar_width, label='Winter')

    ax.set_xlabel('Years')
    ax.set_ylabel('Number of dates for each season (mean per season)')
    ax.set_title('Number of points per season where CASM shap value is {} {} for at least {} days(mean per season)'.format(adverb,boundary,treshold))

    ax.set_xticks(x)
    ax.set_xticklabels(years)

    ax.legend()

    plt.show()
    
def add_lat_lon_season(data_with_casm,predictions_with_casm,predictions_without_casm,predictions_with_casm_and_elevation):
    latitudes=data_with_casm[data_with_casm['start_date'].dt.year==2011]['lat']
    latitudes.reset_index(inplace=True,drop=True)
    longitudes=data_with_casm[data_with_casm['start_date'].dt.year==2011]['lon']
    longitudes.reset_index(inplace=True,drop=True)
    for year in range(2011,2018):
        t=time.time()

        predictions_with_casm[year]['results'].reset_index(inplace=True,drop=True)
        predictions_without_casm[year]['results'].reset_index(inplace=True,drop=True)
        predictions_with_casm_and_elevation[year]['results'].reset_index(inplace=True,drop=True)
        predictions_with_casm[year]['results']['lat']=latitudes
        predictions_with_casm[year]['results']['lon']=longitudes
        predictions_with_casm_and_elevation[year]['results']['lon']=longitudes
        predictions_with_casm_and_elevation[year]['results']['lat']=latitudes
        predictions_without_casm[year]['results']['lat']=latitudes
        predictions_without_casm[year]['results']['lon']=longitudes

        predictions_with_casm[year]['results']['season'] = predictions_with_casm[year]['results']['Date'].apply(assign_season)
        predictions_without_casm[year]['results']['season'] = predictions_without_casm[year]['results']['Date'].apply(assign_season)
        predictions_with_casm_and_elevation[year]['results']['season'] = predictions_with_casm_and_elevation[year]['results']['Date'].apply(assign_season)
        print('Elapsed time=',time.time()-t)
        
def string_climate_region(climate_region):
    if climate_region=='B':
        return 'Arid regions'
    elif climate_region=='Cc':
        return 'Mild temperate continental regions'
    elif climate_region=='Co':
        return 'Mild temperate oceanic regions'
    else:
        return 'Continental climate regions'

def recover_predictions_seasons():
    predictions_with_casm_and_elevation = {            2011: { 
                                                           'results':pd.DataFrame()

              },            
    2012: { 
                       'results':pd.DataFrame()

              },
        2013: { 
                           'results':pd.DataFrame()

              },
    2014: { 
                       'results':pd.DataFrame()

    },
    2015: {
                    'results':pd.DataFrame()

    },
    2016: {
                    'results':pd.DataFrame()

    },
    2017: {
                       'results':pd.DataFrame()

      }
    }
    predictions_with_casm = {2011: { 
                                                       'results':pd.DataFrame()

          },            
    2012: { 
                   'results':pd.DataFrame()

          },
    2013: { 
                       'results':pd.DataFrame()

          },
    2014: { 
                   'results':pd.DataFrame()

    },
    2015: {
                'results':pd.DataFrame()

    },
    2016: {
                'results':pd.DataFrame()

    },
    2017: {
                   'results':pd.DataFrame()

      }
    }
    predictions_without_casm = {            2011: { 
                                                       'results':pd.DataFrame()

          },            
    2012: { 
                   'results':pd.DataFrame()

          },
    2013: { 
                       'results':pd.DataFrame()

          },
    2014: { 
                   'results':pd.DataFrame()

    },
    2015: {
                'results':pd.DataFrame()

    },
    2016: {
                'results':pd.DataFrame()

    },
    2017: {
                   'results':pd.DataFrame()

      }
    }

    for year in range(2011,2018):
        predictions_with_casm_and_elevation[year]['results']=pd.read_hdf('results/predictions/rodeo_casm_elevation_seasons.h5',key='/{}/results'.format(year))
    for year in range(2011,2018):
        predictions_with_casm[year]['results']=pd.read_hdf('results/predictions/rodeo_casm_seasons.h5',key='/{}/results'.format(year))
    for year in range(2011,2018):
        predictions_without_casm[year]['results']=pd.read_hdf('results/predictions/rodeo_only_seasons.h5',key='/{}/results'.format(year))
    return predictions_with_casm_and_elevation,predictions_with_casm,predictions_without_casm

def create_dataframes_for_skills_analysis(predictions_with_casm,predictions_without_casm,predictions_with_casm_and_elevation):
    #this function creates one dataframe per year
    #we create dataframes because it's an object which is easier to handle than dictionnary containing arrays as values
    df2011 = pd.DataFrame()
    df2011['start_date'] = predictions_with_casm[2011]['results']['Date']
    df2011['lat'] = predictions_with_casm[2011]['results']['lat']
    df2011['lon'] = predictions_with_casm[2011]['results']['lon']
    df2011['ground truth'] = predictions_with_casm[2011]['results']['Actual Temperature']
    df2011['preds with casm'] = predictions_with_casm[2011]['results']['Predicted Temperature']
    df2011['preds with casm and elevation'] = predictions_with_casm_and_elevation[2011]['results']['Predicted Temperature']
    df2011['preds without casm'] = predictions_without_casm[2011]['results']['Predicted Temperature']
    df2011['season'] = predictions_with_casm[2011]['results']['season']

    df2012 = pd.DataFrame()
    df2012['start_date'] = predictions_with_casm[2012]['results']['Date']
    df2012['lat'] = predictions_with_casm[2012]['results']['lat']
    df2012['lon'] = predictions_with_casm[2012]['results']['lon']
    df2012['ground truth'] = predictions_with_casm[2012]['results']['Actual Temperature']
    df2012['preds with casm'] = predictions_with_casm[2012]['results']['Predicted Temperature']
    df2012['preds with casm and elevation'] = predictions_with_casm_and_elevation[2012]['results']['Predicted Temperature']
    df2012['preds without casm'] = predictions_without_casm[2012]['results']['Predicted Temperature']
    df2012['season'] = predictions_with_casm[2012]['results']['season']

    
    df2013 = pd.DataFrame()
    df2013['start_date'] = predictions_with_casm[2013]['results']['Date']
    df2013['lat'] = predictions_with_casm[2013]['results']['lat']
    df2013['lon'] = predictions_with_casm[2013]['results']['lon']
    df2013['ground truth'] = predictions_with_casm[2013]['results']['Actual Temperature']
    df2013['preds with casm'] = predictions_with_casm[2013]['results']['Predicted Temperature']
    df2013['preds with casm and elevation'] = predictions_with_casm_and_elevation[2013]['results']['Predicted Temperature']
    df2013['preds without casm'] = predictions_without_casm[2013]['results']['Predicted Temperature']
    df2013['season'] = predictions_with_casm[2013]['results']['season']

    df2014 = pd.DataFrame()
    df2014['start_date'] = predictions_with_casm[2014]['results']['Date']
    df2014['lat'] = predictions_with_casm[2014]['results']['lat']
    df2014['lon'] = predictions_with_casm[2014]['results']['lon']
    df2014['ground truth'] = predictions_with_casm[2014]['results']['Actual Temperature']
    df2014['preds with casm'] = predictions_with_casm[2014]['results']['Predicted Temperature']
    df2014['preds with casm and elevation'] = predictions_with_casm_and_elevation[2014]['results']['Predicted Temperature']
    df2014['preds without casm'] = predictions_without_casm[2014]['results']['Predicted Temperature']
    df2014['season'] = predictions_with_casm[2014]['results']['season']

    df2015 = pd.DataFrame()
    df2015['start_date'] = predictions_with_casm[2015]['results']['Date']
    df2015['lat'] = predictions_with_casm[2015]['results']['lat']
    df2015['lon'] = predictions_with_casm[2015]['results']['lon']
    df2015['ground truth'] = predictions_with_casm[2015]['results']['Actual Temperature']
    df2015['preds with casm'] = predictions_with_casm[2015]['results']['Predicted Temperature']
    df2015['preds with casm and elevation'] = predictions_with_casm_and_elevation[2015]['results']['Predicted Temperature']
    df2015['preds without casm'] = predictions_without_casm[2015]['results']['Predicted Temperature']
    df2015['season'] = predictions_with_casm[2015]['results']['season']

    df2016 = pd.DataFrame()
    df2016['start_date'] = predictions_with_casm[2016]['results']['Date']
    df2016['lat'] = predictions_with_casm[2016]['results']['lat']
    df2016['lon'] = predictions_with_casm[2016]['results']['lon']
    df2016['ground truth'] = predictions_with_casm[2016]['results']['Actual Temperature']
    df2016['preds with casm'] = predictions_with_casm[2016]['results']['Predicted Temperature']
    df2016['preds with casm and elevation'] = predictions_with_casm_and_elevation[2016]['results']['Predicted Temperature']
    df2016['preds without casm'] = predictions_without_casm[2016]['results']['Predicted Temperature']
    df2016['season'] = predictions_with_casm[2016]['results']['season']

    df2017 = pd.DataFrame()
    df2017['start_date'] = predictions_with_casm[2017]['results']['Date']
    df2017['lat'] = predictions_with_casm[2017]['results']['lat']
    df2017['lon'] = predictions_with_casm[2017]['results']['lon']
    df2017['ground truth'] = predictions_with_casm[2017]['results']['Actual Temperature']
    df2017['preds with casm'] = predictions_with_casm[2017]['results']['Predicted Temperature']
    df2017['preds with casm and elevation'] = predictions_with_casm_and_elevation[2017]['results']['Predicted Temperature']
    df2017['preds without casm'] = predictions_without_casm[2017]['results']['Predicted Temperature']
    df2017['season'] = predictions_with_casm[2017]['results']['season']


    return df2011,df2012,df2013,df2014,df2015,df2016,df2017

def restore_dict_dataframes():
    dict_dataframes={}
    with pd.HDFStore('results/predictions/dict_dataframes.h5', mode='r') as store:
        keys = store.keys()
        for key in keys:
            cle = key.lstrip('/') 
            dataframe = store[key]
            year = int(cle.split('_')[1]) 
            dic_test[year] = dataframe
    return dict_dataframes
    
def pointwise_skill_computation(df,year):
    """
    we need to group by latitude and longitude to form batches per season. 
    we then perform skill computation on every batch
    """
    df['skill with casm'] = None
    df['skill without casm']=None
    df['skill with casm and elevation']=None
    df.reset_index(inplace=True,drop=True)
    skills_with_casm=[]
    skills_with_casm_and_elevation=[]
    skills_without_casm=[]

    groups = df.groupby(['lat','lon','season'])

    for _, group in groups:

        skill_with_casm=get_col_skill(group,'preds with casm','ground truth',date_col='start_date')
        skill_with_casm_and_elevation=get_col_skill(group,'preds with casm and elevation','ground truth',date_col='start_date')
        skill_without_casm=get_col_skill(group,'preds without casm','ground truth',date_col='start_date')
        
        skills_with_casm.append(skill_with_casm)
        skills_without_casm.append(skill_without_casm)

        df.loc[group.index, 'skill with casm'] = skill_with_casm
        df.loc[group.index, 'skill without casm'] = skill_without_casm

    return df

def compute_pointwise_skills(dict_dataframes):
    for year in range(2011,2018):
        t=time.time()
        df=dict_dataframes[year].copy()
        df=pointwise_skill_computation(df,year)
        dict_dataframes[year]=df
        print('Elapsed time=',time.time()-t)
    return dict_dataframes

def compute_skills_evolutions(dict_dataframes):
    for year in range(2011,2018):
        t=time.time()
        df=dict_dataframes[year].copy()
        df['skill improvement for casm']=(df['skill with casm']-df['skill without casm'])/df['skill without casm']*100
        df['skill improvement for elevation']=(df['skill with casm and elevation']-df['skill with casm'])/df['skill with casm']*100

        dict_dataframes[year]=df
        print('Elapsed time=',time.time()-t)
    return dict_dataframes

def plot_skills_improvements(dict_dataframes,season,year,feature):
    if feature=='casm':
        season_to_study=season
        year_to_study=year
        df=dict_dataframes[year]


        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.LambertConformal()})

        ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())

        ax.add_feature(cartopy.feature.STATES)

        filtered_df1 = df[(df['skill improvement for casm'] <= 100)&(df['season']==season_to_study)&(df['skill improvement for casm'] >= 10)]
        filtered_df2 = df[(df['skill improvement for casm'] <= -10)&(df['season']==season_to_study)&(df['skill improvement for casm'] >= -100)]
        filtered_df3 = df[(df['skill improvement for casm'] <= 10)&(df['season']==season_to_study)&(df['skill improvement for casm'] >= -10)]



        marker_size=40
        sc1 = ax.scatter(filtered_df1['lon'], filtered_df1['lat'], c=filtered_df1['skill improvement for casm'], marker='s', s=marker_size,cmap='Greens', transform=ccrs.PlateCarree())
        sc2 = ax.scatter(filtered_df2['lon'], filtered_df2['lat'], c=filtered_df2['skill improvement for casm'],marker='s', s=marker_size, cmap='Reds', transform=ccrs.PlateCarree())
        sc3 = ax.scatter(filtered_df3['lon'], filtered_df3['lat'], c=filtered_df3['skill improvement for casm'],marker='s', s=marker_size, cmap='Greys', transform=ccrs.PlateCarree())



        cbar1 = plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.04)
        cbar1.set_label('Skill improvement')
        cbar2 = plt.colorbar(sc2, ax=ax, fraction=0.03, pad=0.04)
        cbar2.set_label('Skill improvement')
        cbar3 = plt.colorbar(sc3, ax=ax, fraction=0.03, pad=0.04)
        cbar3.set_label('Skill improvement')

        ax.set_title('Skills improvements with CASM feature for {} in {}'.format(year_to_study,string_season(season_to_study)))

        plt.show()
    if feature=='elevation':
        season_to_study=season
        year_to_study=year
        df=dict_dataframes[year]


        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.LambertConformal()})

        ax.set_extent([-125, -66.5, 20, 50], ccrs.Geodetic())

        ax.add_feature(cartopy.feature.STATES)

        filtered_df1 = df[(df['skill improvement for elevation'] <= 100)&(df['season']==season_to_study)&(df['skill improvement for elevation'] >= 10)]
        filtered_df2 = df[(df['skill improvement for elevation'] <= -10)&(df['season']==season_to_study)&(df['skill improvement for elevation'] >= -100)]
        filtered_df3 = df[(df['skill improvement for elevation'] <= 10)&(df['season']==season_to_study)&(df['skill improvement for elevation'] >= -10)]



        marker_size=40
        sc1 = ax.scatter(filtered_df1['lon'], filtered_df1['lat'], c=filtered_df1['skill improvement for elevation'], marker='s', s=marker_size,cmap='Greens', transform=ccrs.PlateCarree())
        sc2 = ax.scatter(filtered_df2['lon'], filtered_df2['lat'], c=filtered_df2['skill improvement for elevation'],marker='s', s=marker_size, cmap='Reds', transform=ccrs.PlateCarree())
        sc3 = ax.scatter(filtered_df3['lon'], filtered_df3['lat'], c=filtered_df3['skill improvement for elevation'],marker='s', s=marker_size, cmap='Greys', transform=ccrs.PlateCarree())



        cbar1 = plt.colorbar(sc1, ax=ax, fraction=0.03, pad=0.04)
        cbar1.set_label('Skill improvement')
        cbar2 = plt.colorbar(sc2, ax=ax, fraction=0.03, pad=0.04)
        cbar2.set_label('Skill improvement')
        cbar3 = plt.colorbar(sc3, ax=ax, fraction=0.03, pad=0.04)
        cbar3.set_label('Skill improvement')

        ax.set_title('Skills improvements with elevation feature for {} in {}'.format(year_to_study,string_season(season_to_study)))

        plt.show()

def skills_modifications_bars(dict_dataframes,df_climate_zones,area):
        
    for year in range(2011,2018):
        df=dict_dataframes[year]
        columns_to_drop = [col for col in df.columns if col.startswith('Large')]
        df = df.drop(columns=columns_to_drop)
        columns_to_drop = [col for col in df.columns if col.startswith('p1901')]
        df = df.drop(columns=columns_to_drop)

        df=df.merge(df_climate_zones[['lat','lon','Large region']],on=['lat','lon'],how='left')

        df['Large region'] = df.apply(transform_large_region, axis=1)
        dict_dataframes[year]=df

    num_rows = 2
    num_cols = 4
    area_to_study=area
    limit=10
    points=len(dict_dataframes[2011][dict_dataframes[2011]['Large region'] == area_to_study]['skill improvement for casm'].unique())

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 8))
    fig.suptitle('Repartition of skills modifications in {} for years 2011-2017  ({} points)'.format(string_climate_region(area_to_study),points), fontsize=16)
    fig.subplots_adjust(hspace=0.4)

    positive_means=[]
    useless_means=[]
    negative_means=[]
    for i, year_to_study in enumerate(range(2011, 2018)):
        df=dict_dataframes[year_to_study]

        filtered_df1 = df[(df['Large region'] == area_to_study) & (df['skill improvement for casm'] >= limit)]
        filtered_df2 = df[(df['skill improvement for casm'] <= -limit) & (df['Large region'] == area_to_study) ]
        filtered_df3 = df[(df['skill improvement for casm'] <= limit) & (df['Large region'] == area_to_study) & (df['skill improvement for casm'] >= -limit)]

        lengths = [len(filtered_df1['skill improvement for casm'].unique()), len(filtered_df2['skill improvement for casm'].unique()), len(filtered_df3['skill improvement for casm'].unique())]
        total=lengths[0]+lengths[1]+lengths[2]
        percentage=[lengths[0]/total*100,lengths[1]/total*100,lengths[2]/total*100]
        positive_means.append(percentage[0])
        useless_means.append(percentage[1])
        negative_means.append(percentage[2])

        labels = ['Positive skills \nimprovements\n(>{}%)'.format(limit), 'Useless points', 'Negative skills \nimprovements\n(<-{}%)'.format(limit)]

        row = i // num_cols
        col = i % num_cols

        axes[row, col].bar(labels, percentage)

        for j, percentage in enumerate(percentage):
            axes[row, col].text(j, percentage, '{:.1f}'.format(percentage), ha='center', va='bottom')

        axes[row, col].set_ylabel('Number of points')
        axes[row, col].set_title('Year {}'.format(year_to_study))

    percentage=[statistics.mean(positive_means),statistics.mean(useless_means),statistics.mean(negative_means)]
    labels = ['Positive skills \nimprovements\n(>{}%)'.format(limit), 'Useless points', 'Negative skills \nimprovements\n(<-{}%)'.format(limit)]

    axes[-1, -1].bar(labels,percentage )
    for j, percentage in enumerate(percentage):
        axes[-1, -1].text(j, percentage, '{:.1f}'.format(percentage), ha='center', va='bottom')

    axes[-1, -1].set_ylabel('Number of points')
    axes[-1, -1].set_title('Global (2011-2017)')


    plt.tight_layout()

    plt.show()
    

def scatter_plot(data_with_elevation_and_casm,dict_dataframes,shap_values_with_casm_and_elevation,year,season):
    dataframe_indices=pd.read_hdf('results/shap_values/dataframe_indices.h5')
    casm_data=data_with_elevation_and_casm[data_with_elevation_and_casm['start_date'].dt.year>=2011][['lat','lon','start_date','CASM']]
    casm_data.sort_values(by=['start_date','lat','lon'],inplace=True)
    casm_data.reset_index(inplace=True,drop=True)
    casm_data['index']=dataframe_indices['index']
    casm_data['season']=dataframe_indices['season']
    casm_data['day']=dataframe_indices['day']
    casm_data['year']=casm_data['start_date'].dt.year
    
    test_casm=casm_data.groupby(['lat','lon','year','season'],as_index=False)['CASM'].mean()

    year_to_study=year
    season_to_study=season

    df=dict_dataframes[year]

    dfinter=df[df['season']==season_to_study][['lat','lon','skill improvement for casm']]
    testcasmplot=test_casm[(test_casm['year']==year_to_study)&(test_casm['season']==season_to_study)][['lat','lon','CASM']]
    dfplot=pd.DataFrame()
    dfplot['lat']=testcasmplot['lat']
    dfplot['lon']=testcasmplot['lon']
    dfplot['skill improvement for casm']=dfinter['skill improvement for casm'].unique()

    mask = np.array([sub_array[13] == season_to_study for sub_array in shap_values_with_casm_and_elevation[year_to_study]])
    shap_values_season=shap_values_with_casm_and_elevation[year_to_study][mask]

    numpy_index = shap_values_season[:, 12]

    unique_numpy_index= np.unique(numpy_index)

    means = []
    for index in unique_numpy_index:
        mask = (numpy_index == index)  
        group = shap_values_season[mask] 
        casm_shap_mean = np.mean(group[:, 10])  
        means.append(casm_shap_mean)

    casm_shap_mean= pd.DataFrame({'Index': unique_numpy_index, 'CASM shap values mean': means})

    scatterplot=dfplot.copy()
    scatterplot['CASM']=testcasmplot['CASM']
    scatterplot['index']=dataframe_indices['index'].unique()
    scatterplot.reset_index(inplace=True,drop=True)
    scatterplot['CASM shap values mean']=casm_shap_mean['CASM shap values mean']



    filtered_scatterplot=scatterplot[(abs(scatterplot['skill improvement for casm'])<=100)&(scatterplot['skill improvement for casm']>=-100)&(abs(scatterplot['skill improvement for casm'])>=10)&(scatterplot['CASM']<=.4)]

    filtered_scatterplot['CASM'] = pd.to_numeric(filtered_scatterplot['CASM'], errors='coerce')
    filtered_scatterplot['skill improvement for casm'] = pd.to_numeric(filtered_scatterplot['skill improvement for casm'], errors='coerce')

    filtered_scatterplot = filtered_scatterplot.dropna(subset=['CASM', 'skill improvement for casm'])

    x=filtered_scatterplot['CASM']
    y=filtered_scatterplot['skill improvement for casm']

    coefficients = np.polyfit(x, y, 2)

    curve_x = np.linspace(min(x), max(x), 100)
    curve_y = np.polyval(coefficients, curve_x) 

    point_colors = filtered_scatterplot['CASM shap values mean']


    plt.scatter(filtered_scatterplot['CASM'], filtered_scatterplot['skill improvement for casm'], c=point_colors, cmap='RdYlGn')

    plt.plot(curve_x, curve_y, color='red', label='Quadratic regression')


    plt.xlabel('CASM')
    plt.ylabel('Skill Improvement')

    # Ajouter un titre
    plt.title('Scatter Plot: CASM vs Skill Improvement in {} {}'.format(string_season(season_to_study),year_to_study))
    plt.legend()

    cbar = plt.colorbar()
    cbar.set_label('CASM shap values mean')
    plt.show()
