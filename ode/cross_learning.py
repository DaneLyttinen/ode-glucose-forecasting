import argparse
import itertools
import os
import hashlib
from sklearn.preprocessing import RobustScaler
from ode.Participant import Participant
import pickle
import numpy as np
import pandas as pd
import shutil
from neuralforecast.models import LSTM, NHITS, RNN, TFT, DilatedRNN, TCN
from neuralforecast.losses.pytorch import RMSE, MAPE, MQLoss
from lightning.pytorch.accelerators import find_usable_cuda_devices
import torch
import random
from sklearn.decomposition import PCA
from neuralforecast import NeuralForecast
import utilsforecast.processing as ufp
from ode.utils import get_directory_names
## if any datasets do not contain static vars
non_static_vars = []
## if there are any columns, or states, which need to be disregarded.
cols_to_discard = []
# clusters of keys and their respective cluster group if performing SHAP
clusters = {'AA12345': 1}

def init_models(run_name, horizon, state_columns, loss, devices):
    max_steps = 750
    exog_type = 'stat_exog_list' if state_columns and ("lipids" in run_name or "metabolites" in run_name) else 'hist_exog_list'
    if devices == None:
        try:
            free_gpus = [device_id for device_id in range(torch.cuda.device_count()) if torch.cuda.utilization(device_id) == 0]
            free_gpus = [x for x in free_gpus if x != 3]
            devices   = [free_gpus[0]]
        except:
            devices = [0]
    else:
        devices = [devices]
    model_params = {
        'LSTM': {'h': horizon, 'max_steps': max_steps, 'scaler_type': 'standard', 'encoder_hidden_size': 64, 
                 'decoder_hidden_size': 64, 'loss': loss, "accelerator" : "cuda", "devices" : devices, "logger" : False},
        'NHITS': {'h': horizon, 'input_size': 8, 'max_steps': max_steps, 'n_freq_downsample': [2, 1, 1], 'loss': loss, "accelerator" : "cuda", "devices" : devices, "logger" : False},
        'RNN': {'h': horizon, 'max_steps': max_steps, 'loss': loss, "accelerator" : "cuda", "devices" : devices, "logger" : False},
        'TFT': {'h': horizon, 'max_steps': max_steps, 'loss': loss, 'input_size': 8, "accelerator" : "cuda", "devices" : devices, "logger" : False},
        'DilatedRNN': {'h': horizon, 'max_steps': max_steps, 'loss': loss, "accelerator" : "cuda", "devices" : devices, "logger" : False},
        'TCN': {'h': horizon, 'max_steps': max_steps, 'loss': loss, "accelerator" : "cuda", "devices" : devices, "logger" : False},
    }
    
    if state_columns:
        for model in model_params:
            model_params[model][exog_type] = state_columns
    
    models = [
        LSTM(**model_params['LSTM']),
        NHITS(**model_params['NHITS']),
        RNN(**model_params['RNN']),
        TFT(**model_params['TFT']),
        DilatedRNN(**model_params['DilatedRNN']),
        TCN(**model_params['TCN']),
    ]
    
    return models


def generate_hash(argument):
    return hashlib.md5(argument.encode()).hexdigest()[:8]

def prepare_dataset(df, run_name, all_participants, dimension):
    master_df = df
    master_df["time"] = pd.to_datetime(master_df['time'], format="%Y-%m-%d %H:%M:%S")
    master_df = master_df.rename(columns={"gl": "y", "time": "ds"})
    static_df = None
    cols_to_drop = None
    state_columns = None
    if "meal" in run_name:
        state_columns = ["meal"]
        cols_to_remove = [col for col in master_df.columns if col not in ["ds", "y", "meal", "unique_id"]]
        master_df = master_df.drop(columns=cols_to_remove)
    elif "multivariate" in run_name:
        state_columns = [col for col in master_df.columns if col.startswith('state') and col not in cols_to_discard]
        cols_to_drop = [col for col in master_df.columns if col.startswith('state') and col in cols_to_discard]
        master_df.drop(columns=cols_to_drop, inplace=True)
    elif "lipids" in run_name or "metabolites" in run_name:
        if len(all_participants.static_vars.keys()) == 1:
            if dimension != None:
                state_columns = [f"PC{x}" for x in range(1,dimension +1)]
            else:    
                key = list(all_participants.static_vars.keys())[0]
                state_columns = all_participants.static_vars[key].tolist()
            to_drop = [col for col in master_df.columns if col not in ["ds", "y", "meal", "unique_id"]]
            to_drop.append("meal")
        else:
            state_columns = [col for col in master_df.columns if col not in ["ds", "y", "meal", "unique_id"]]
            to_drop = ["meal"]
            to_drop.extend(state_columns)
        static_df = master_df[state_columns + ["unique_id"]]
        master_df = master_df.drop(columns=to_drop)
    else:
        cols_to_remove = [col for col in master_df.columns if col not in ["ds", "y", "unique_id"]]
        master_df = master_df.drop(columns=cols_to_remove)
    return master_df,static_df,state_columns, cols_to_drop

def test(hold_out_datasets, hold_out_key, run_name, cols_to_drop, nf, modelSave, horizon, loss, prefix, hold_out_participant, train_static):
    for datasetOrig in hold_out_datasets:
        datasetName = datasetOrig.name
        dataset = datasetOrig.copy(deep=True)
        test_set_size = (round((len(dataset) / 100) * 20))
        val_set_size = (round((len(dataset) / 100) * 10))

        dataset["unique_id"] = hold_out_key[2:]

        #static_df = create_eval_static(run_name, dimension, dataset, test_set_size)
        static_df = None
        if isinstance(train_static, pd.DataFrame):
            static_df = train_static[train_static["unique_id"] == float(datasetName[2:].replace("_", "."))]
            static_df.reset_index(drop=True, inplace=True)
        dataset['gl'] = dataset['gl'].apply(lambda value: round(value * 18, 1))
        dataset["time"] = pd.to_datetime(dataset['time'], format="%Y-%m-%d %H:%M:%S")
        dataset = dataset.rename(columns={"gl": "y", "time": "ds"})
        if "multivariate" in run_name:
            #cols_to_drop = [col for col in master_df.columns if col.startswith('state') and col in cols_to_discard]
            dataset.drop(columns=cols_to_drop, inplace=True)
        dataset.name = datasetName
        cv_df = cross_validate(dataset, test_set_size, nf, static_df, modelSave,horizon)
        hold_out_participant.saveData(prefix, dataset, cv_df, horizon, loss)

def main():
    # Argument parser for hold-out key and other configurations
    global non_static_vars
    global cols_to_discard
    parser = argparse.ArgumentParser(description="Cross learning with hold-out participant")
    parser.add_argument('--hold-out-key',required=True, type=str, help="Hold-out participant key")
    parser.add_argument("--run_name", "--rn", default="cluster_cross_learning", type=str)
    parser.add_argument("--error-index", "--ei", default=1, type=int)
    parser.add_argument("--dimension", default=None, type=int)
    parser.add_argument("--remove-series", default="None", type=str, help="Series to remove by unique_id")
    parser.add_argument("--horizon", "--h", default=2, type=int)
    parser.add_argument("--overwrite", default=False, type=bool)
    parser.add_argument("--cluster", "--c", default=False, type=bool)
    parser.add_argument("--devices", default=None, type=int)
    
    args = parser.parse_args()
    idx = args.error_index
    prefix = "cross_learning"
    error_metrics = [MAPE, RMSE, MQLoss]
    horizon = args.horizon
    error = error_metrics[idx]

    levels = [80,90]
    loss = error(level=levels) if idx == 2 else error()
    run_name = args.run_name 


    max_val_size = 0
    max_test_size = 0
    all_dfs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hold_out_key = args.hold_out_key  # Use the key provided via argparse
    if (hold_out_key == "AA12345"):
        print("skipping AA12345")
        return
    if (hold_out_key in non_static_vars and ("lipids" in run_name or "metabolites" in run_name)):
        return
    keys = get_directory_names(script_dir)
    if ("lipids" in run_name or "metabolites" in run_name):
        keys = [key for key in keys if key not in non_static_vars]

    hold_out_participant = Participant(hold_out_key, None, run_name)

    all_run_name = "multivariate" if "multivariate" in run_name else run_name
    all_participants = Participant("AA12345", None, all_run_name)
    df = all_participants.getDataFrames()[0]

    if args.remove_series != "None":
        hash_value = generate_hash(args.remove_series)
        prefix += f"_remove_{hash_value}"
        series_to_remove = args.remove_series.split(',')
        df = df[~df['unique_id'].isin(series_to_remove)]
        print(df["unique_id"].unique())
    train_static = None
    if ("lipids" in run_name or "metabolites" in run_name) and args.dimension != None:
        
        columns = list(all_participants.static_vars.values())
        columns = [x.tolist() for x in columns]
        flattened_list = [item for sublist in columns for item in sublist]
    
        #train_static = df[flattened_list]

        static_features = df[flattened_list]
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(static_features)
        pca = PCA(n_components=args.dimension)
        reduced_features = pca.fit_transform(scaled_features)

        reduced_df = pd.DataFrame(reduced_features, columns=[f'PC{i+1}' for i in range(args.dimension)])
        df = pd.concat([df[['unique_id', 'meal', 'time', 'gl']], reduced_df], axis=1)
        print(df.head())
        prefix += f"_dim{args.dimension}"

 
        train_static = df[df['unique_id'].astype(str).str.startswith(hold_out_key[2:])]
        train_static = train_static.drop(columns=["meal", "time","gl"])
        train_static.reset_index(drop=True, inplace=True)
    
    if ("lipids" in run_name or "metabolites" in run_name) and args.dimension == None:
        columns = list(all_participants.static_vars.values())
        columns = [x.tolist() for x in columns]
        flattened_list = [item for sublist in columns for item in sublist]

        static_features = df[flattened_list + ["unique_id"]]
        train_static = static_features[static_features['unique_id'].astype(str).str.startswith(hold_out_key[2:])]
        train_static.reset_index(drop=True, inplace=True)
        #train_static.drop(columns=["unique_id"])

    if "pretrain" in run_name:
        reduced_series_list = []
        hold_out_series = df[df['unique_id'].astype(str).str.startswith(hold_out_key[2:])]
        original_length = len(df)
        for unique_id in hold_out_series['unique_id'].unique():
            temp_series = df[df['unique_id'] == unique_id]
            reduced_length = int(len(temp_series) * 0.7)
            reduced_series = temp_series.iloc[:reduced_length]
            reduced_series_list.append(reduced_series)
        reduced_series_list.append(df[~df['unique_id'].astype(str).str.startswith(hold_out_key[2:])])
        df = pd.concat(reduced_series_list, ignore_index=True)
        assert original_length > len(df)
        assert len(hold_out_series) > 0
        
        print(df.head())
    else:
        df = df[~df['unique_id'].astype(str).str.startswith(hold_out_key[2:])]
    # Reset the index to ensure it is continuous
    df.reset_index(drop=True, inplace=True)

    hold_out_datasets = hold_out_participant.getDataFrames()

    def filter_runs(dataset):
        return not hold_out_participant.checkDataExists(prefix, dataset, None, horizon, loss)
    if not args.overwrite:
        hold_out_datasets = list(filter(filter_runs, hold_out_datasets))

    if len(hold_out_datasets) == 0:
        print(f"Skipping key {hold_out_participant.Id}")
    elif(args.cluster):
        cluster_indices = set(clusters.values())
        for r in range(1,7):
            combinations = list(itertools.combinations(cluster_indices, r))
            # Get unique_ids for the specified cluster
            for cluster_combination in combinations:
                unique_ids = [float(uid[2:].replace("_", ".")) for uid, cidx in clusters.items() if cidx in cluster_combination]
                cluster_master_df = df[df['unique_id'].isin(unique_ids)]
                cluster_str = '_'.join(map(str, cluster_combination))

                prefix = f"cluster_cross_learning_{cluster_str}"

                cluster_master_df,static_df,state_columns, cols_to_drop = prepare_dataset(cluster_master_df, run_name, all_participants, args.dimension)
                cluster_master_df.name = f"{cluster_str}"

                models = init_models(run_name,horizon, None, loss, args.devices)
                nf = NeuralForecast(models=models, freq='15min', local_scaler_type=None)
                nf.fit(cluster_master_df)

                modelSave = os.path.join(all_participants.path, f"{hold_out_key}_"+all_participants.getSaveString(prefix, cluster_master_df, horizon, loss))
                nf.save(path=modelSave, model_index=None, overwrite=True, save_dataset=False)
                test(hold_out_datasets, hold_out_key, run_name, None, nf, modelSave, horizon, loss, prefix, hold_out_participant, train_static)
    else:
        master_df,static_df,state_columns, cols_to_drop = prepare_dataset(df, run_name, all_participants, args.dimension)
        master_df.name = run_name
        models = init_models(run_name,horizon, state_columns, loss, args.devices)
        modelSave = os.path.join(all_participants.path, f"{hold_out_key}_"+all_participants.getSaveString(prefix, master_df, horizon, loss))
        if (not os.path.isdir(modelSave) or args.overwrite):
            nf = NeuralForecast(models=models, freq='15min', local_scaler_type=None)
            if "lipids" in run_name or "metabolites" in run_name:
                nf.fit(master_df, static_df=static_df)
            else:
                nf.fit(master_df)
            nf.save(path=modelSave, model_index=None, overwrite=True, save_dataset=False)
        else:
            nf = NeuralForecast.load(path=modelSave)
        test(hold_out_datasets, hold_out_key, run_name, cols_to_drop, nf, modelSave, horizon, loss, prefix, hold_out_participant,train_static)
    print(f"Cross learning completed for {hold_out_participant.Id}")
    shutil.rmtree(modelSave)

def cross_validate(dataset, test_set_size, nf, static_df, modelSave, horizon):
    h = horizon
    step_size = 1
    id_col = 'unique_id'
    time_col = 'ds'
    target_col = 'y'

    n_windows = int((test_set_size - h) / step_size) + 1
    splits = ufp.backtest_splits(
        dataset,
        n_windows=n_windows,
        h=h,
        id_col=id_col,
        time_col=time_col,
        freq="15min",
        step_size=step_size,
        input_size=None,
    )
    results = []
    nf = NeuralForecast.load(path=modelSave)
    for i_window, (cutoffs, train, test) in enumerate(splits):
        
        predict_df = train
        preds = nf.predict(
            df=predict_df,
            static_df=static_df,
        )
        preds = ufp.join(preds, cutoffs, on=id_col, how="left")
        fold_result = ufp.join(
            preds, test[[id_col, time_col, target_col]], on=[id_col, time_col]
        )
        results.append(fold_result)
    out = ufp.vertical_concat(results, match_categories=False)
    out = ufp.drop_index_if_pandas(out)
    # match order of cv with no refi
    # t
    first_out_cols = [id_col, time_col, "cutoff"]
    remaining_cols = [
        c for c in out.columns if c not in first_out_cols + [target_col]
    ]
    cols_order = first_out_cols + remaining_cols + [target_col]
    out = ufp.sort(out[cols_order], by=[id_col, "cutoff", time_col])

    return out

if __name__ == "__main__":
    main()
