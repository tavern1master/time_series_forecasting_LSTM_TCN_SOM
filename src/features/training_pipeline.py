from functools import partial
import os
import json
import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d, Module, Sequential, LSTM, Dropout, ReLU
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

import json
import numpy as np

from models.tcn import TCN
from models.lstm import AutoregressiveLSTM
from data.make_dataset import WindowDataset

EPOCHS = 30

def train_tcn(config, data_dir, verbose=True):
    model = TCN(input_size=1, 
                    output_size=config['target_size'],
                    seq_length=config['seq_length'],
                    num_channels=[1],
                    kernel_size=config['kernel_size'],
                    dropout=config['dropout'])
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    dset = WindowDataset(data_dir['DATA'], config["seq_length"], config["target_size"], mode=config['mode'])
    train_len, test_len = int(0.7 * len(dset)), int(0.15 * len(dset))
    train_dset, test_dset, val_dset = random_split(dset, [train_len, test_len, len(dset) - train_len - test_len])

    train_loader = DataLoader(train_dset, config["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dset, config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dset, config["batch_size"], shuffle=True, drop_last=True)

    
    for epoch in range(start_epoch, EPOCHS):
        running_loss = 0.0
        epoch_steps = 0
        model.init_weights()
        
        model.train()
        for counter, (x, y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = model.forward(x)

            loss = criterion(y_hat, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if counter % 2000 == 1999:
                print(
                    "[%d, %5d] loss: %.3f"
                            % (epoch + 1, counter + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        val_loss = 0.0
        val_steps = 0

        model.eval()
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)

            y_hat = model.forward(x)

            loss = criterion(y_hat, y)
            # val_loss += loss.cpu().numpy()
            val_loss += loss.cpu().detach().numpy()
            val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        checkpoint.to_directory(os.path.join(data_dir['CHECKPOINT'], f"checkpoint{epoch}.pkl"))
        session.report(
            {"loss": val_loss / val_steps},
            checkpoint=checkpoint,)

    print("Finished Training")

def train_lstm(config, data_dir, verbose=True):
    model = AutoregressiveLSTM(input_size=1, 
                             hidden_size=config['hidden_size'],
                             n_layers=config['n_layers'],
                             dropout=config['dropout'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    dset = WindowDataset(data_dir['DATA'], config["seq_length"], config["target_size"])
    train_len, test_len = int(0.7 * len(dset)), int(0.15 * len(dset))
    train_dset, test_dset, val_dset = random_split(dset, [train_len, test_len, len(dset) - train_len - test_len])

    train_loader = DataLoader(train_dset, config["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dset, config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dset, config["batch_size"], shuffle=True, drop_last=True)

    for epoch in range(start_epoch, EPOCHS):
        running_loss = 0.0
        epoch_steps = 0
        h_t, c_t = model.init_hidden(config["batch_size"])
        h_t, c_t = h_t.to(device), c_t.to(device)

        model.train()
        for counter, (x, y) in enumerate(train_loader):
            x,y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat, (h_t, c_t) = model.forecast(x, (h_t.data, c_t.data), config["target_size"])

            loss = criterion(y_hat, y)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if counter % 2000 == 1999:
                print(
                    "[%d, %5d] loss: %.3f"
                            % (epoch + 1, counter + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        val_loss = 0.0
        val_steps = 0

        h_t, c_t = model.init_hidden(config["batch_size"])
        h_t, c_t = h_t.to(device), c_t.to(device)

        model.eval()
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)

            y_hat, (h_t, c_t) = model.forecast(x, (h_t.data, c_t.data), config["target_size"])

            loss = criterion(y_hat, y)
            # val_loss += loss.cpu().numpy()
            val_loss += loss.cpu().detach().numpy()
            val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)
        
        os.makedirs(data_dir['CHECKPOINT'],exist_ok = True)
        checkpoint.to_directory(os.path.join(data_dir['CHECKPOINT'], f"checkpoint{epoch}.pkl"))
        session.report(
            {"loss": val_loss / val_steps},
            checkpoint=checkpoint,)

    print("Finished Training")

def test_tcn_accuracy(model, data_dir, config, device="cpu"):
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

#     return np.mean(val_losses), val_losses, forecast, test_vals
    return np.mean(val_losses)

def test_lstm_accuracy(model, data_dir, config, device="cpu"):
    dset = WindowDataset(data_dir['DATA'], config["seq_length"], config["target_size"])
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

#     return np.mean(val_losses), val_losses, forecast, test_vals
    return np.mean(val_losses)

def main_tcn(num_samples=20, max_num_epochs=30, gpus_per_trial=0):
    # Change CLUSTER to conduct training on another dataset
    CLUSTER = 'df'
    DATEINAME = CLUSTER+".csv"

    PFAD2DATA = r"..\data\processed" if "notebook" in os.getcwd() else "data\processed"
    PFAD2CHECKPOINT = r"..\models\checkpoints" if "notebook" in os.getcwd() else "models\checkpoints"
    PFAD2RESULTS = r"..\models\results" if "notebook" in os.getcwd() else "models\results"

    PATH2DATA = os.path.join(os.getcwd(), PFAD2DATA, DATEINAME)
    PATH2CHECKPOINT = os.path.join(os.getcwd(), PFAD2CHECKPOINT, CLUSTER)
    PATH2RESULTS = os.path.join(os.getcwd(), PFAD2RESULTS, CLUSTER)
    PATH = os.getcwd()

    data_dir = {
        'DATA': PATH2DATA,
        'CHECKPOINT': PATH2CHECKPOINT,
        'RESULTS': PATH2RESULTS,
        'WORKDIR': PATH
    }

    # Define config with hyperparameters
    config = {
        "seq_length": tune.choice([2, 4, 6, 8, 10, 12, 24, 48]), # Choose one of 4 options: month, quartal, half a year, year
        "target_size": tune.sample_from(lambda spec: np.random.randint(1, spec.config.seq_length)),
        "kernel_size": tune.choice([2, 4, 8, 16, 32]),
        "dropout": tune.choice([0.1, 0.2, 0.25, 0.3, 0.4]),
        "lr": tune.loguniform(1e-2, 1),
        "batch_size": tune.choice([4, 8, 16, 32, 64, 128, 256]),
        "mode": "tcn"
    } 
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2, 
    )

    result = tune.run(
        partial(train_tcn, data_dir=data_dir),
        config=config,
        num_samples = num_samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial}
    )

    best_trial = result.get_best_trial("loss","min","last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    os.makedirs(os.path.join(data_dir['RESULTS']),exist_ok = True)
    with open(os.path.join(data_dir['RESULTS'], f"best_trial_config_tcn.json"), 'w') as fp:
        json.dump(best_trial.config, fp)
    
    best_trained_model = TCN(
                input_size=1, 
                output_size=best_trial.config['target_size'],
                seq_length=best_trial.config['seq_length'],
                num_channels=[1],
                kernel_size=best_trial.config['kernel_size'],
                dropout=best_trial.config['dropout']
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    
    os.makedirs(os.path.join(data_dir['RESULTS']),exist_ok = True)
    torch.save(best_trained_model.state_dict(), os.path.join(data_dir['RESULTS'], f"best_trained_tcn.pkl"))
    
    test_acc = test_tcn_accuracy(best_trained_model, data_dir, best_trial.config, device)
    print("Best trial test set accuracy: {}".format(test_acc))

def main_lstm(num_samples=8, max_num_epochs=20, gpus_per_trial=0):
    # Change CLUSTER to conduct training on another dataset
    CLUSTER = 'df'
    DATEINAME = "cluster_7.csv"

    PFAD2DATA = r"..\data\processed" if "notebook" in os.getcwd() else "data\processed"
    PFAD2CHECKPOINT = r"..\models\checkpoints" if "notebook" in os.getcwd() else "models\checkpoints"
    PFAD2RESULTS = r"..\models\results" if "notebook" in os.getcwd() else "models\results"

    PATH2DATA = os.path.join(os.getcwd(), PFAD2DATA, DATEINAME)
    PATH2CHECKPOINT = os.path.join(os.getcwd(), PFAD2CHECKPOINT, CLUSTER)
    PATH2RESULTS = os.path.join(os.getcwd(), PFAD2RESULTS, CLUSTER)
    PATH = os.getcwd()

    data_dir = {
        'DATA': PATH2DATA,
        'CHECKPOINT': PATH2CHECKPOINT,
        'RESULTS': PATH2RESULTS,
        'WORKDIR': PATH
    }

    # Define config with hyperparameters
    config = {
        "seq_length": tune.choice([10, 12, 24, 48]),
        "target_size": 1,
        "hidden_size": tune.qrandint(96, 154, 12),
        "n_layers":  2,
        "dropout": tune.choice([0.1, 0.2, 0.25, 0.3]),
        "lr": tune.loguniform(1e-2, 1),
        "batch_size": tune.choice([8, 16, 32, 64, 128, 256]),
        "mode": "lstm"
    } 

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2, 
    )

    result = tune.run(
        partial(train_lstm, data_dir=data_dir),
        config=config,
        num_samples = num_samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial}
    )

    best_trial = result.get_best_trial("loss","min","last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
#     print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")
    
    os.makedirs(data_dir['RESULTS'],exist_ok = True)
    with open(os.path.join(data_dir['RESULTS'], "best_trial_config_lstm.json"), 'w') as fp:
        json.dump(best_trial.config, fp)

    best_trained_model = AutoregressiveLSTM(
        input_size=1, 
        hidden_size=best_trial.config['hidden_size'],
        n_layers=best_trial.config['n_layers'],
        dropout=best_trial.config['dropout']
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if gpus_per_trial > 1:
        best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])
    
    os.makedirs(data_dir['RESULTS'],exist_ok = True)
    torch.save(best_trained_model.state_dict(), os.path.join(data_dir['RESULTS'], f"best_trained_lstm.pkl"))
    
    test_acc = test_lstm_accuracy(best_trained_model, data_dir, best_trial.config, device)
    print("Best trial test set accuracy: {}".format(test_acc))