from src.import_data.import_data import import_data
from src.prepare_data.prepare_data import prepare_data_creation, merge_data
from src.create_model.create_model import create_model
from src.analyze_data.analyze_data import analyze_data


def insurance_weather_model(country, bool_log=True, bool_return=True, bool_all_gdp=True, bool_classification=True):
    """
    Funktion liest daten zu einem Land ein
    """
    # Import Data
    data, geo_data = import_data(country)

    # prepare each data individually
    data = prepare_data_creation(
        data, bool_return=bool_return, bool_log=bool_log)

    # analyze original data
    analyze_data(data)

    # merge different datasets
    data = merge_data(data=data, geo_data=geo_data, bool_all_gdp=bool_all_gdp)

    # create model
    create_model(data, 'GDP_Value', bool_classification=bool_classification)
