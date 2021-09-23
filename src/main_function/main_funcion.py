from src.import_data.import_data import import_data
from src.prepare_data.prepare_data import prepare_data

from src.analyze_data.analyze_data import analyze_data


def create_model(country):
    """
    Funktion liest daten zu einem Land ein
    """
    data = import_data(country)
    data = prepare_data(data, bool_return=True, bool_log = True)
    analyze_data(data)
    print(data.head())