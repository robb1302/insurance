from src.config.settings import years as YEARS
import matplotlib.pyplot as plt
import seaborn as sns
import src.utils.globals as GLOBALS
import numpy as np

def analyze_data(data):
    years_names = [str(y)for y in YEARS]
    matrix = np.round(data[years_names].transpose().corr(),3)
    matrix.to_csv(GLOBALS.OUT_FOLDER+'Corr.csv',sep=';',decimal=',')
    #plt.figure(figsize=(100,100))
    #sns.heatmap(matrix, vmin=0, vmax=1, cmap="Blues", linewidths=0.5, annot=True)
    #plt.show()