import os
import numpy as np
import pandas as pd
import json
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import random_split, DataLoader

from src.models.tcn import TCN
from src.models.lstm import AutoregressiveLSTM
from src.data.make_dataset import WindowDataset
from src.config import Config



def create_data_dir_dict(mode:str, cluster:str):
    '''
    Supporting function that organizes directories.
    Mode is LSTM or TCN
    Cluster is name of the training dataset
    Returns dictionary with Pfads to the dataset, working directory and model's section in results.
    If empy mode or empty cluster is provided, returns working directory
    '''
    MODEL = mode
    CLUSTER = cluster
    DATEINAME = CLUSTER+".csv"
    PFAD2DATA = r"..\data\processed" if "notebook" in os.getcwd() else r"data\processed"
    PFAD2CHECKPOINT = r"..\models\checkpoints" if "notebook" in os.getcwd() else r"models\checkpoints"
    PFAD2RESULTS = r"..\models\results" if "notebook" in os.getcwd() else r"models\results"

    PATH2DATA = os.path.join(os.getcwd(), PFAD2DATA, DATEINAME)
    PATH2CHECKPOINT = os.path.join(os.getcwd(), PFAD2CHECKPOINT, MODEL, CLUSTER)
    PATH2RESULTS = os.path.join(os.getcwd(), PFAD2RESULTS, MODEL, CLUSTER)
    PATH = os.getcwd()

    data_dir = {
            'DATA': PATH2DATA,
            'CHECKPOINT': PATH2CHECKPOINT,
            'RESULTS': PATH2RESULTS,
            'WORKDIR': PATH
        }
    return data_dir

def test_tcn_accuracy(model, data_dir, config, mean=True, device="cpu"):
    '''
    Calculates MSE on the dataset provided in data_dir.
    Config is a set of parameters resulted in parameter optimization.
    Returns array of MSE. Returns mean MSE if mean equals True
    '''
    dset = WindowDataset(data_dir['DATA'], config["seq_length"], config["target_size"], mode=config['mode'])
    train_len, test_len = int(0.7 * len(dset)), int(0.15 * len(dset))
    train_dset, test_dset, val_dset = random_split(dset, [train_len, test_len, len(dset) - train_len - test_len])

    test_loader = DataLoader(test_dset, config["batch_size"], shuffle=True, drop_last=True)

    val_losses = []
    criterion = torch.nn.MSELoss()
    
    model.eval()
    for counter, (x,y) in enumerate(test_loader):
        x,y = x.to(device), y.to(device)

        y_hat = model.forward(x)
        
        val_loss = criterion(y_hat, y)
        val_losses.append(val_loss.item())

    if mean:
        return np.mean(val_losses)
    return val_losses

def test_lstm_accuracy(model, data_dir, config, mean=True, device="cpu"):
    '''
    see test_tcn_accuracy()
    '''
    dset = WindowDataset(data_dir['DATA'], config["seq_length"], config["target_size"])
    train_len, test_len = int(0.7 * len(dset)), int(0.15 * len(dset))
    train_dset, test_dset, val_dset = random_split(dset, [train_len, test_len, len(dset) - train_len - test_len])

    test_loader = DataLoader(test_dset, config["batch_size"], shuffle=True, drop_last=True)

    val_losses = []
    criterion = torch.nn.MSELoss()
    
    h_t, c_t = model.init_hidden(config["batch_size"])
    h_t, c_t = h_t.to(device), c_t.to(device)
    
    model.eval()
    for counter, (x,y) in enumerate(test_loader):
        x,y = x.to(device), y.to(device)

        y_hat, (h_t, c_t) = model.forecast(x, (h_t.data, c_t.data), config["target_size"])
        
        val_loss = criterion(y_hat, y)
        val_losses.append(val_loss.item())

    if mean:
        return np.mean(val_losses)
    return val_losses

def overview_of_configurations(data_dir:dict, mode:str):
    '''
    Reads configs saved in results folder after parameter optimization
    '''
    path2folders = os.path.join(data_dir['WORKDIR'], 'models/results/'+mode)
    dict2folders = {}
    for subdir, dirs, files in os.walk(path2folders):
        for file in files:
            key = os.path.basename(subdir) + os.path.splitext(file)[1]
            dict2folders[key] = os.path.join(subdir, file)
    
    dict2configs = {}
    for key, val in dict2folders.items():
        if ".json" in key:
            p2json = val
            new_key = key.split('.')[0]
            with open(p2json, "rb") as fp:
                config = json.load(fp)
            dict2configs[new_key] = config
            
    return dict2configs

def stability_test(data_dir:dict, mode:str, mean=True):
    '''
    Reads all configurations, declares models.
    Conducts stability test on dataset from data_dir
    Returns dictionary with models, dictionary with configs and dictionary with MSE on validation dataset
    '''
    path2folders = os.path.join(data_dir['WORKDIR'], 'models/results/'+mode)
    dict2folders = {}
    for subdir, dirs, files in os.walk(path2folders):
        for file in files:
            key = os.path.basename(subdir) + os.path.splitext(file)[1]
            dict2folders[key] = os.path.join(subdir, file)
            
    dict2models = {}
    for key, val in dict2folders.items():
        p2json = None
        if ".pkl" in key:
            p2pkl = val
        if ".json" in key:
            p2json = val
        if p2pkl and p2json:
            new_key = key.split('.')[0]
            with open(p2json, "rb") as fp:
                config = json.load(fp)
            if mode == 'tcn':
                model = TCN(input_size=1, 
                        output_size=config['target_size'],
                        seq_length=config['seq_length'],
                        num_channels=[1],
                        kernel_size=config['kernel_size'],
                        dropout=config['dropout'])
            elif mode == 'lstm':
                model = AutoregressiveLSTM(
                        input_size=1, 
                        hidden_size=config['hidden_size'],
                        n_layers=config['n_layers'],
                        dropout=config['dropout'])
                
            model.load_state_dict(torch.load(p2pkl))
            dict2models[new_key] = model
            
    dict2configs = {}
    for key, val in dict2folders.items():
        if ".json" in key:
            p2json = val
            new_key = key.split('.')[0]
            with open(p2json, "rb") as fp:
                config = json.load(fp)
            dict2configs[new_key] = config
            
    dict2loss = {}
    for k,m in dict2models.items():
        if mode == 'tcn':
            dict2loss[k] = test_tcn_accuracy(m, data_dir, dict2configs[k], mean=mean)
        else:
            dict2loss[k] = test_lstm_accuracy(m, data_dir, dict2configs[k], mean=mean)
        
    return dict2models, dict2configs, dict2loss

def experiment_2():
    '''
    Conducts stability test for all models on every cluster
    '''
    result_dict = {}
    MODE = ['tcn', 'lstm']
    for m in MODE:
        res = {}
        # Conduct test on clusters
        for i in range(8):
            clname = f'cluster_{str(i)}'
            ddir = create_data_dir_dict(mode=m, cluster=clname)
            d1, d2, d3 = stability_test(ddir, mode=m, mean=True)
            res[clname] = d3
        # Conduct test on the entire dataset
        clname = f'df'
        ddir = create_data_dir_dict(mode=m, cluster=clname)
        d1, d2, d3 = stability_test(ddir, mode=m, mean=True)
        res[clname] = d3
            
        result_dict[m] = res
    return result_dict

def experiment_stability_test():
    res_dict = experiment_2()
    tcn = pd.DataFrame(res_dict['tcn'])
    min_MSE_tcn = tcn.min().min()
    max_MSE_tcn = tcn.max().max()
    lstm = pd.DataFrame(res_dict['lstm'])
    min_MSE_lstm = lstm.min().min()
    max_MSE_lstm = lstm.max().max()
    tcn.to_csv(Config.get_path2file('stability_test_tcn.csv', tables=True))
    lstm.to_csv(Config.get_path2file('stability_test_lstm.csv', tables=True))
    print(f'LSTM min MSE : {min_MSE_lstm} | LSTM max MSE: {max_MSE_lstm}')
    print(f'TCN min MSE: {min_MSE_tcn} | TCN max MSE: {max_MSE_tcn}')

def get_models_and_configs(data_dir:dict, mode:str):
    '''
    Declare models and read configs for future experiments
    '''
    path2folders = os.path.join(data_dir['WORKDIR'], 'models/results/'+mode)
    dict2folders = {}
    for subdir, dirs, files in os.walk(path2folders):
        for file in files:
            key = os.path.basename(subdir) + os.path.splitext(file)[1]
            dict2folders[key] = os.path.join(subdir, file)
            
    dict2models = {}
    for key, val in dict2folders.items():
        p2json = None
        if ".pkl" in key:
            p2pkl = val
        if ".json" in key:
            p2json = val
        if p2pkl and p2json:
            new_key = key.split('.')[0]
            with open(p2json, "rb") as fp:
                config = json.load(fp)
            if mode == 'tcn':
                model = TCN(input_size=1, 
                        output_size=config['target_size'],
                        seq_length=config['seq_length'],
                        num_channels=[1],
                        kernel_size=config['kernel_size'],
                        dropout=config['dropout'])
            elif mode == 'lstm':
                model = AutoregressiveLSTM(
                        input_size=1, 
                        hidden_size=config['hidden_size'],
                        n_layers=config['n_layers'],
                        dropout=config['dropout'])
                
            model.load_state_dict(torch.load(p2pkl))
            dict2models[new_key] = model
            
    dict2configs = {}
    for key, val in dict2folders.items():
        if ".json" in key:
            p2json = val
            new_key = key.split('.')[0]
            with open(p2json, "rb") as fp:
                config = json.load(fp)
            dict2configs[new_key] = config
            
    return dict2models, dict2configs

def test_forecast_accuracy(data_dir:dict, mode:str, fixed=None, predict=False):
    '''
    Calculates MSE of all models on the benchmark
    '''
    dict2models, dict2configs = get_models_and_configs(data_dir, mode=mode)
    benchmark = pd.read_csv(data_dir['DATA'], index_col=0, parse_dates=True)
    # Normalization
    benchmark = benchmark.to_numpy(dtype=np.float32)
    mean = np.mean(benchmark, axis=1, keepdims=True)
    std = np.std(benchmark, axis=1, keepdims=True)
    idx = np.where(std)[0]
    benchmark, mean, std = benchmark[idx], mean[idx], std[idx]
    benchmark = (benchmark - mean)/std
    
    # Evaluation metric
    criterion = torch.nn.MSELoss()
    # Save results in dict
    results_dict = {}
    if predict:
            predictions = []
    # Forecasting loop
    names = dict2models.keys()
    for name in names:
        # Model declaration
        model = dict2models[name]
        # Benchmark slicing
        fwindow = dict2configs[name]['seq_length']
        fhorizon = dict2configs[name]['target_size']
        if fixed is not None:
            fhorizon = fixed
        upper_bound = len(benchmark)-fwindow-fhorizon
        lower_bound = len(benchmark)-fhorizon
        X = torch.from_numpy(benchmark[upper_bound:lower_bound].T).float()
        Y = torch.from_numpy(benchmark[lower_bound:].T).float()
        # X size ([61, seq_length]) -> X[0] is of size [seq_length]
        # Y size ([61, target_size]) -> Y[0] is of size [target_size]
        
        # Setup LSTM
        if mode == 'lstm':
            h_t, c_t = model.init_hidden(batch_size=1)
    
        # Create forecast for each benchmark time series as save loss
        forecasting_loss = []
        model.eval()
        for i, x in enumerate(X):
            y = Y[i]
            if mode == 'tcn':
                x = x.view((1,1,x.shape[0]))
                y_hat = model.forecast(x, fhorizon)
            if mode == 'lstm':
                x = x.view((1,x.shape[0],1))
                y_hat = model.forecast_evaluation(x, (h_t,c_t), fhorizon)
#                 y_hat = y_hat.squeeze()
            loss = criterion(y_hat, y)
            forecasting_loss.append(loss.item())
            if predict:
                predictions.append({'forecast':y_hat.detach().numpy(), 'original data': y.detach().numpy()})
        results_dict[name] = forecasting_loss
    
    if predict:
        return results_dict, predictions
    return results_dict 

def experiment3(fixed, predict=False):
    '''
    Automate forecasting accuracy calculation
    Fixed is forecasting horizon. If none, parameters from config applied by default
    '''
    modes = ['lstm', 'tcn']
    data = 'benchmark'
    
    results = {}
    if predict:
        predictions = {}
        
    for m in modes:
        ddir = create_data_dir_dict(m, data)
        if predict:
            results[m], predictions[m] = test_forecast_accuracy(ddir, m, fixed=fixed, predict=predict)
        else:
            results[m] = test_forecast_accuracy(ddir, m, fixed=fixed, predict=predict)
    if predict:
        return results, predictions
    return results

def apply_auto_arima(data, seasonal=False):
    # Convert data to a pandas Series if it's not already
    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    # Find the best ARIMA order using auto_arima
    stepwise_fit = auto_arima(data, seasonal=seasonal, stepwise=True, suppress_warnings=True)

    # Get the best ARIMA order
    p, d, q = stepwise_fit.order

    # Fit the ARIMA model with the best order
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(data, order=(p, d, q))
    fitted_model = model.fit()

    return fitted_model

def apply_auto_arima_and_predict(df, x, seasonal=False):
    predictions = pd.DataFrame()
    mse_results = {}

    for col in df.columns:
        data = df[col].dropna()  # Remove any missing values
        training_data = data.iloc[:-x]
        testing_data = data.iloc[-x:]

        # Find the best ARIMA order using auto_arima
        stepwise_fit = auto_arima(training_data, seasonal=seasonal, stepwise=True, suppress_warnings=True)
        p, d, q = stepwise_fit.order

        # Fit the ARIMA model with the best order
        model = ARIMA(training_data, order=(p, d, q))
        fitted_model = model.fit()

        # Make predictions
        predictions[col] = fitted_model.forecast(steps=x)

        # Calculate MSE for the last 'x' values
        mse = mean_squared_error(testing_data, predictions[col])
        mse_results[col] = mse

    return predictions, mse_results

def normalize_df(df):
    data = df.values
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    idx = np.where(std)[0]
    data, std, mean = data[idx], std[idx], mean[idx]
    return (data - mean)/std

def experiment_forecast_accuracy():
    # Forecasting horizon [from Parameter optimization, weekly, quarterly, halt-term, full year]
    fhorizon = [None, 1, 12, 24, 48]
    for fh in fhorizon:
        gfm_result = experiment3(fh)
        for m in ["lstm", 'tcn']:
            res_df = pd.DataFrame(gfm_result[m])
            res_df.to_csv(Config.get_path2file(f'experiment_forecasting_accuracy_{m}_fh_{fh}.csv', tables=True))

    bench = pd.read_csv(Config.get_path2file('benchmark.csv', processeddata=True), index_col=0)
    bench = normalize_df(bench)
    benchmark = pd.DataFrame(bench)
    arima_result = {}
    for fh in fhorizon:
        if fh is not None:
            _, mse = apply_auto_arima_and_predict(benchmark, fh)
            arima_result[fh] = mse
    arima_df = pd.DataFrame(arima_result)
    arima_df.to_csv(Config.get_path2file(f'experiment_forecasting_accuracy_ARIMA.csv', tables=True))

    
    


