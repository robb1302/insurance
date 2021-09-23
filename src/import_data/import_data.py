import pandas as pd
import src.utils.globals as GLOBALS
import matplotlib.pyplot as plt
import numpy as np
import ppscore as pps
import seaborn as sns
pd.set_option('max_columns',500)

def import_data():
    data = pd.read_csv('data\SriLanka\API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2916517.csv',encoding="utf8")
    data = data.rename(columns={'value':'GDP_Value'})

    years_names = set(data.columns)-set(['Country Name', 'Country Code','Unnamed: 65','Indicator Name','Indicator Code'])
    years_change_names = [i+'_change_abs' for i in list(years_names)]
    years_change_names.sort()
    return data