import argparse
import os
from ode.Participant import Participant
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NHITS, RNN, StemGNN, TFT, PatchTST, DilatedRNN, TCN
import traceback
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import *
import pandas as pd
from neuralforecast.losses.pytorch import RMSE, MSE, MQLoss, MAPE
from neuralforecast.auto import *
import re

minimum = 100000
has_been_optimized = False
params_arr = []

def generate_result_columns(models, horizon):
    columns = ['time'] + [f'actual_t+{i}' for i in range(horizon)]
    for model in models:
        model_name = model.__class__.__name__
        columns += [f'{model_name}_pred_t+{i}' for i in range(horizon)] + [f'{model_name}_error']
    return columns


def init_models(run_name, horizon, state_columns, loss):
    max_steps = 750
    if state_columns != None:
        models = [
                LSTM(h=horizon, max_steps=max_steps, scaler_type='standard', encoder_hidden_size=64, 
                        decoder_hidden_size=64, loss=loss, hist_exog_list=state_columns),    
                NHITS(h=horizon, input_size=8, max_steps=max_steps, n_freq_downsample=[2, 1, 1], 
                        loss=loss, hist_exog_list=state_columns),
                RNN(h=horizon, max_steps=max_steps, loss=loss, hist_exog_list=state_columns),
                TFT(h=horizon, max_steps=max_steps, loss=loss, hist_exog_list=state_columns, input_size=8),
                DilatedRNN(h=horizon, max_steps=max_steps, loss=loss, hist_exog_list=state_columns),
                TCN(h=horizon, max_steps=max_steps, loss=loss, hist_exog_list=state_columns)
                ]
    else:
        models = [
                #StemGNN(n_series=n_series, h=horizon, max_steps=max_steps, input_size=20)
                LSTM(h=horizon, max_steps=max_steps, scaler_type='standard', encoder_hidden_size=64, 
                        decoder_hidden_size=64, loss=loss),
                NHITS(h=horizon, input_size=8, max_steps=max_steps, n_freq_downsample=[2, 1, 1], 
                        loss=loss),
                RNN(h=horizon, max_steps=max_steps, loss=loss),
                TFT(h=horizon, max_steps=max_steps, loss=loss, input_size=8),
                DilatedRNN(h=horizon, max_steps=max_steps, loss=loss),
                TCN(h=horizon, max_steps=max_steps, loss=loss)
                ]
    return models

def create_neuralforecast(run_name, models):
    if "scaler" in run_name:
        nf = NeuralForecast(models=models, freq='15min', local_scaler_type="robust-iqr")
    else:
        nf = NeuralForecast(models=models, freq='15min')
    return nf

def fit_auto_models(dataset, run_name, horizon, loss):
    global params_arr
    global has_been_optimized
    modelClasses = [LSTM ,NHITS,RNN, TFT, DilatedRNN, TCN]
    if ("auto" in run_name and not has_been_optimized):
        autoModels = [AutoLSTM(h=horizon,loss=loss, num_samples=15),
                AutoNHITS(h=horizon, loss=loss, num_samples=15),
                AutoRNN(h=horizon, loss=loss, num_samples=15),
                AutoTFT(h=horizon, loss=loss, num_samples=15),
                AutoDilatedRNN(h=horizon, loss=loss, num_samples=15),
                AutoTCN(h=horizon, loss=loss, num_samples=15)]
        trainSize = (round((len(dataset) / 100) * 80))
        validation_size = (round((len(dataset) / 100) * 10))
        validationDataset = dataset[:trainSize]
        nf = create_neuralforecast(run_name, autoModels)
        #training_set = merged_dataset[:current_index+1]
        nf.fit(df=validationDataset, val_size=validation_size)
        models = []
        for i in range(len(autoModels)):
            best_params = nf.models[i].results.get_best_result( 
    metric="loss", mode="min").config
            params_arr.append(best_params)
            models.append(modelClasses[i](**best_params))
        has_been_optimized = True


def fit_model(dataset, run_name, horizon, state_columns, loss):
    global has_been_optimized
    global params_arr
    models = init_models( run_name, horizon, state_columns, loss)
    if ("auto" in run_name and has_been_optimized):
        models = []
        modelClasses = [LSTM ,NHITS,RNN, TFT, DilatedRNN, TCN]
        for i in range(len(modelClasses)):
            best_params = params_arr[i]
            models.append(modelClasses[i](**best_params))
    nf = create_neuralforecast(run_name, models)
    nf.fit(df=dataset)
    return nf

def list_of_strings(arg):
    return arg.split(',')

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def get_directory_names(path):
    # List all directories in the given path
    directories = [d for d in os.listdir(path) 
                   if os.path.isdir(os.path.join(path, d)) and d != '__pycache__']
    return directories

def grid_search(dataset, participant, key, run_name,horizon,loss):
    df_copy = dataset.copy(deep=True)
    df_copy = df_copy.rename(columns={"gl": "y", "time": "ds"})
    df_copy.name = dataset.name
    if participant.checkDataExists(f"best_params", df_copy, params_arr, horizon, loss):
        print(f"Skipping participant {key}")
        return
    df_copy["unique_id"] = 1
    fit_auto_models(df_copy, run_name, horizon, loss)
    participant.saveData(f"best_params", df_copy, params_arr, horizon, loss)

def train(error, keys, run_name, horizon):
    is_univariate = "univariate" in run_name

    for key in keys:
        #error = #error_metrics[idx]
        loss = error
        
        participant = Participant(key, run_name)
        datasets = participant.getDataFrames()
        for dataset in datasets:
            if participant.checkDataExists(f"cross_val",dataset, None,horizon, loss):
                continue
            datasetName = dataset.name
            test_set_size = (round((len(dataset) / 100) * 20))
            val_set_size = (round((len(dataset) / 100) * 10))
                #test_set_size = (round((len(dataset) / 100) * 15))
            train_set_size = len(dataset) - test_set_size
            dataset['gl'] = dataset['gl'].apply(lambda value: round(value * 18, 1))
            if ("augment" in run_name):
                match = re.search(r'#(\d+)_', dataset.name)
                train_set_size = int(match.group(1))
                test_set_size = len(dataset) - train_set_size
            dataset["time"] = pd.to_datetime(dataset['time'], format="%Y-%m-%d %H:%M:%S")

            state_columns = [col for col in dataset.columns if col.startswith('state')] if not is_univariate else ["meal"] if "meal" in run_name else None
            print(state_columns)
            if ("auto" in run_name):
                grid_search(dataset, participant, key, run_name,horizon,loss)
                continue
            models = init_models( run_name, horizon, state_columns, loss)
            nf = create_neuralforecast(run_name, models)
            dataset["unique_id"] = 1.0
            
            dataset = dataset.rename(columns={"gl": "y", "time": "ds"})
            dataset.name = datasetName
            cv_df = nf.cross_validation(dataset,n_windows=None, val_size=val_set_size, test_size=test_set_size, refit=5)
            participant.saveData(f"cross_val",dataset, cv_df,horizon, loss)

# It is expected to have a foldername of a regex defined in Participant, with CSV files starting with the folder name
# Alongside columns gl and time, for including meal, meal columns is needed
# The states generated by the ODE should be in the CSV if multivariate is in the run_name, prefixed with "state"
# The state generation, not covered here, needs to be done in an online fashion and inserted to the csv ending with multivariate
# run_name - acc_rand = 50% accurate meal estimation, acc = 75% accurate meal estimation
#          - multivariate = ODE + gl
#          - meal = gl + meal
def main():
    global has_been_optimized
    global params_arr
    # need to run
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", "--rn", default="multivariate_scaler_second", type=str)
    parser.add_argument('--keys',required=True, type=list_of_strings)
    parser.add_argument("--horizon", "--h", default=2, type=int)
    parser.add_argument("--error-index", "--ei", default=None, type=int)
    args = parser.parse_args()

    keys = list(filter(None, args.keys))
    run_name = args.run_name
    horizon = args.horizon
    levels = [80,90]
    error_metrics = [MAPE, RMSE, MQLoss]
    idx = args.error_index
    if idx == None:
        for idx, error in enumerate(error_metrics):
            loss = error(level=levels) if idx == 2 else error()
            train(loss, keys, run_name, horizon)
    else:
        error = error_metrics[idx]
        loss = error(level=levels) if idx == 2 else error()
        train(loss, keys, run_name, horizon)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An exception occurred: {str(e)}")
        traceback.print_exc()
        exit(1)
