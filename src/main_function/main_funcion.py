from src.import_data.import_data import import_data
from src.prepare_data.prepare_data import prepare_data,merge_data

from src.analyze_data.analyze_data import analyze_data


def create_model(country):
    """
    Funktion liest daten zu einem Land ein
    """
    data,geo_data = import_data(country)
    data = prepare_data(data, bool_return=True, bool_log = True)
    analyze_data(data)
    data = merge_data(data = data,geo_data=geo_data)
    print(data.head())