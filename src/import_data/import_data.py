import pandas as pd
import src.utils.globals as GLOBALS
import matplotlib.pyplot as plt
import numpy as np
import ppscore as pps
import seaborn as sns
import geojson
pd.set_option('max_columns', 500)


def import_data(country):
    """
    """
    data = pd.read_csv(GLOBALS.DATA_FOLDER+country +
                       '\API_NY.GDP.PCAP.CD_DS2_en_csv_v2_2916517.csv', encoding="utf8")
    data = data.rename(columns={'value': 'GDP_Value'})
    

    path_to_file = GLOBALS.DATA_FOLDER+"\Historical_Tsunami_Event_Locations.geojson"
    with open(path_to_file, encoding="cp437") as f:
        gj = geojson.load(f)
    features = gj['features'][0]

    return data,gj
