from src.data.transform_rawdata import data_preparation
from src.experiments.experiments import experiment_stability_test, experiment_forecast_accuracy

def main():
    data_preparation()
    experiment_stability_test()
    experiment_forecast_accuracy()