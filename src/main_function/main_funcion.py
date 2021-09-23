from src.import_data.import_data import import_data

def create_model():
    data = import_data()
    print(data.head())