import pandas as pd
import numpy as np
from src.config.settings import years as YEARS


def prepare_data_creation(data, bool_return=False, bool_log=False):
    """
    prepares data for modelling and analysis
    """

    print('Start: Preparation')
    years_names = [str(y) for y in YEARS]
    data[years_names] = log_data(data=data[years_names], bool_log=bool_log)

    data = return_value_data(
        data=data, years_names=years_names, bool_return=bool_return)

    data = data.drop(['Unnamed: 65'], axis=1, errors='ignore')

    # NA mit mean
    data = fix_na_data(data=data, years_names=years_names)

    data = data.set_index("Country Name",drop=False)

    return data


def fix_na_data(data, years_names):
    fixed_data_years = data[years_names].apply(
        lambda x: x.fillna(x.mean()), axis=1)
    data = data.drop(years_names, axis=1, errors='ignore')
    data = pd.concat([data, fixed_data_years], axis=1)
    return data


def log_data(data, bool_log):
    """
    loa dataset
    """
    if bool_log:
        data = np.log(data)
    return data


def return_value_data(data, years_names, bool_return):
    """
    calcualtes return data
    """

    if bool_return:
        # create columns names
        years_change_names = [i+'_change_abs' for i in list(years_names)]
        years_change_names.sort()
        # Create diff
        data_change_gdp = data[years_names].transpose(
        ).sort_index().transpose().diff(axis=1)
        data_change_gdp.columns = years_names
        # drop yearnames
        data = data.drop(years_names, errors='ignore',axis=1)
        # add new data to data
        data = pd.concat([data, data_change_gdp], axis=1)
    else:
        data[years_names].transpose().sort_index().transpose()
    return data

def merge_data(data,geo_data, bool_all_gdp = False):
    """
    merge GDP und weather data  
    """    
    # Unpivot Jahres Werte
    properties_tsunami = pd.DataFrame([f["properties"] for f in geo_data['features']])
    # unpivot
    melt_gdp = pd.melt(data,id_vars=["Country Name","Country Code","Indicator Name","Indicator Code"])
    melt_gdp = melt_gdp.rename(columns={"variable": "YEAR", "Country Name": "COUNTRY",'value':'GDP_Value'})

    # set index weather data
    properties_tsunami.index=properties_tsunami['COUNTRY'].apply(lambda x:x.lower().replace(' ','_'))+'__'+ properties_tsunami['YEAR'].apply(lambda x:str(x))

    # set index gdp data
    melt_gdp.index=melt_gdp['COUNTRY'].apply(lambda x:x.lower().replace(' ','_'))+'__'+ melt_gdp['YEAR'].apply(lambda x:str(x))


    if bool_all_gdp:
        #keep all gdp data and merge tsunami where possible
        df_tsunami_properties_gdp = melt_gdp.join(properties_tsunami,rsuffix='_delete')
    else:
        #keep all wether data and merge gdp where possible
        df_tsunami_properties_gdp = properties_tsunami.join(melt_gdp,rsuffix='_delete')

    # Choose only month gdp
    df_tsunami_properties_gdp = df_tsunami_properties_gdp[(df_tsunami_properties_gdp['YEAR'].astype('int')>1959) &(df_tsunami_properties_gdp['YEAR'].astype('int')<2021)]

    # ignore not matched data
    # TODO   
    #df_tsunami_properties_gdp = df_tsunami_properties_gdp.drop(df_tsunami_properties_gdp.index[df_tsunami_properties_gdp.GDP_Value.isna()])
    return df_tsunami_properties_gdp