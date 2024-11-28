import argparse
from datetime import datetime
import os
from darts import TimeSeries, concatenate
from sklearn.decomposition import PCA
from ode.Participant import Participant
import numpy as np
import pandas as pd
from darts.models import LightGBMModel
from darts.models import XGBModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics.metrics import mape, mql, rmse
from sklearn.preprocessing import RobustScaler
from ode.utils import get_directory_names
## use SHAP
## Finding a metric to cross learn between specific groups
non_lipids = []
import copy
cols_to_discard = []

def init_models(run_name, horizon, loss):
    lags = 8
    random_state = 1

    # Base parameters for the models
    model_params = {
        'lags': lags,
        'output_chunk_length': horizon,
        'random_state': random_state,
        "multi_models" : False
    }

    # Modify the parameters based on the run_name

    if "lipids" in run_name or "metabolites" in run_name:
        model_params['use_static_covariates'] = True
    else:
        model_params['use_static_covariates'] = False
    if "multivariate" in run_name or "meal" in run_name:
        model_params['lags_past_covariates'] = lags

    model_params_GBM = copy.deepcopy(model_params)

    if loss.__name__ == "mql":
        model_params["quantiles"] = [0.05,0.1, 0.15,0.2,0.25, 0.5,0.75, 0.8,0.85, 0.9, 0.95]
        model_params["likelihood"] = "quantile"
        model_params_GBM = copy.deepcopy(model_params)
        model_params["output_chunk_length"] = 1
    else:
        model_params_GBM["objective"] = loss.__name__
        model_params["eval_metric"] = loss.__name__
    # Define the models using the common and conditional parameters
    models = [
        XGBModel(**model_params),
        LightGBMModel(**model_params_GBM)
    ]
    
    return models

def process_static_covars(run_name, all_participants, dimension, df):
    if ("lipids" in run_name or "metabolites" in run_name):
        
        columns = list(all_participants.static_vars.values())
        columns = [x.tolist() for x in columns]
        flattened_list = [item for sublist in columns for item in sublist]
        
        if dimension != None:
            static_features = df[flattened_list]
            scaler = RobustScaler()
            scaled_features = scaler.fit_transform(static_features)
            pca = PCA(n_components=dimension)
            reduced_features = pca.fit_transform(scaled_features)
            reduced_df = pd.DataFrame(reduced_features, columns=[f'PC{i+1}' for i in range(dimension)])
        else:
            reduced_df = df[flattened_list]

        df = pd.concat([df[['unique_id', 'meal', 'time', 'gl']], reduced_df], axis=1)
        print(df.head())
    return df

def process_historical_covars(run_name, df, dataframe_names, hold_out_key):
    columns_to_use = []
    if "meal" in run_name:
        columns_to_use += ["meal"]
    if "multivariate" in run_name:
        columns_to_use += [col for col in df.columns if col.startswith('state') and col not in cols_to_discard]
    
    if (len(columns_to_use) == 0):
        return [],[]
    #columns_to_use += ["unique_id", "time"]
    data = df[columns_to_use + ["unique_id", "time"]]

    hold_out_covars = []
    past_covariates = []
    for idx,datasetName in enumerate(dataframe_names):
        uid = datasetName
        rows = data[data["unique_id"] == uid]
        rows = rows.drop(columns=["unique_id"])
        rows.reset_index(drop=True, inplace=True)
        cov_ts = TimeSeries.from_dataframe(rows,time_col="time", value_cols=columns_to_use)
        if not str(datasetName).startswith(hold_out_key[2:]):
            past_covariates.append(cov_ts)
        else:
            if "pretrain" in run_name:
                reduced_length = int(len(cov_ts) * 0.7)
                train_ts,test_ts = cov_ts.split_after(reduced_length)
                past_covariates.append(train_ts)
            hold_out_covars.append(cov_ts)
        
    return past_covariates,hold_out_covars

def is_more_than_two_digits(value):
    try:
        return int(value) >= 10
    except ValueError:
        return False

def get_all_series(prefix, run_name, hold_out_key, dimension, univariate):
    all_participants = Participant("AA12345", None, prefix)
    #train_series_list: list = all_participants.loadData(prefix+"_train", None, 2, rmse)

    all_run_name = "multivariate" if "multivariate" in run_name else run_name
    all_participants = Participant("AA12345", None, all_run_name)
    df = all_participants.getDataFrames()[0]
    assert df["gl"].apply(is_more_than_two_digits).all(), f"Not all values in column gl are more than 2 digits."
    df = process_static_covars(run_name, all_participants, dimension, df)

    dataframe_names = df['unique_id'].unique().tolist()
    train_series_list = []
    idx_to_remove = []
    univariate_df = df[["unique_id", "gl", "time"]]
    hold_out_test = []
    for idx,datasetName in enumerate(dataframe_names):
        specific_dataframe = univariate_df[univariate_df["unique_id"] == datasetName]
        specific_dataframe.reset_index(drop=True, inplace=True)
        specific_dataframe.drop(columns=["unique_id"], inplace=True)
        ts = TimeSeries.from_dataframe(specific_dataframe, time_col="time", value_cols=["gl"])
        if ("lipids" in run_name or "metabolites" in run_name):
            #datasetName = train_series_list

            #train_static = df[flattened_list]
            train_static = df.drop(columns=['meal', 'time', 'gl'])
            train_static = train_static[train_static['unique_id'] == datasetName]
            train_static.reset_index(drop=True, inplace=True)
            train_static.drop(columns=["unique_id"], inplace=True)
            single_row = train_static.iloc[[0]]
            ts = ts.with_static_covariates(single_row)

        datasetName = str(datasetName)
        if not datasetName.startswith(hold_out_key[2:]):
            train_series_list.append(ts)
        else:
            if "pretrain" in run_name:
                reduced_length = int(len(ts) * 0.7)
                train_ts,test_ts = ts.split_after(reduced_length)
                train_series_list.append(train_ts)
            hold_out_test.append(ts)
    if not univariate:
        if "pretrain" in run_name:
            assert len(train_series_list) == len(dataframe_names)
        else:
            assert len(train_series_list) < len(dataframe_names)
    # train_series_list = new_train_series_list
    past_covariates,hold_out_covars = process_historical_covars(run_name,df,dataframe_names, hold_out_key)
    #for ts in train_series_list:
    if len(past_covariates) > 0:
        assert len(past_covariates) == len(train_series_list)
    
    assert len(hold_out_test) > 0
    print(f"Number of series {len(train_series_list)}")
    print(f"Number of past covariates {len(past_covariates)}")
    return hold_out_test,hold_out_covars,past_covariates,train_series_list

def get_cutoff(loss,hold_out_participant, prefix, df, horizon):
    lossName = "MQLoss" if loss.__name__ == "mql" else "RMSE"
    dataframe = hold_out_participant.loadData(prefix.replace("_ml", ""), df, horizon, lossName)
    dataframe = dataframe.reset_index(drop=True)
    return dataframe.loc[0,"cutoff"]

def cross_learn(run_name, keys, hold_out_key, loss, horizon,dimension, level=None, univariate=False):
    if ("lipids" in run_name or "metabolites" in run_name):
        keys = [key for key in keys if key not in non_lipids]

    hold_out_participant = Participant(hold_out_key, None, run_name)
    prefix = "cross_learning_ml"
    #covariates_series_list = []

    hold_out_test,hold_out_covars,covariates_series_list, train_series_list = get_all_series(prefix,run_name,hold_out_key, dimension,univariate)
    hold_out_test = sorted(hold_out_test, key=lambda ts: ts.start_time())
    if dimension != None:
        prefix += f"_dim{dimension}"
    num_samples = 100 if loss.__name__ == "mql" else 1
    # Create and train the model on combined datasets
    if "multivariate" in run_name or "meal" in run_name:
        assert len(train_series_list) == len(covariates_series_list)
        assert len(hold_out_covars) == len(hold_out_test)
    models = init_models(run_name, horizon, loss)
    for model in models:
        my_model = model
        if not "univariate":
            print(f"fitting")
            if len(covariates_series_list) > 0:
                my_model.fit(train_series_list, past_covariates=covariates_series_list)
            else:
                my_model.fit(train_series_list)
            print("Model has been fit")
        # Prepare the hold-out participant's data
        dfs = hold_out_participant.getDataFrames()
        dfs = sorted(dfs, key=lambda df: df.loc[0, "time"])
        for idx,ts in enumerate(hold_out_test):
            df = dfs[idx]
            covariate = None
            if (len(hold_out_covars) > 0):
                covariate = hold_out_covars[idx]
            if univariate:
                target_cols = ["gl"]
                df['gl'] = df['gl'].apply(lambda value: round(value * 18, 1))
                prefix = "cross_val_ml"
                ts = TimeSeries.from_dataframe(df, time_col="time", value_cols=target_cols)

            # Define train/test split for the hold-out participant train=70,val=10,test=20
            val_set_size = int(len(ts) * 0.8)
            val_cutoff = df.iloc[val_set_size]["time"]
            val_cutoff = pd.Timestamp(datetime.strptime(str(val_cutoff), "%Y-%m-%d %H:%M:%S"))
            try:
                val_cutoff = get_cutoff(loss, hold_out_participant, prefix,df,horizon)
                #if not univariate:
                val_cutoff = val_cutoff + pd.Timedelta(minutes=15)
            except:
                raise Exception("could not get val cutoff from results df")

            # no_train, test = ts.split_after(val_cutoff)
            if univariate:
                # train_set_size = int(len(df) * 0.7)
                # train_cutoff = df.iloc[train_set_size]["time"]
                # train_cutoff = pd.Timestamp(datetime.strptime(str(train_cutoff), "%Y-%m-%d %H:%M:%S"))
                train,test = ts.split_before(val_cutoff)
                
                # Scale the series using only the training data
               # hold_out_covars = None if len(hold_out_covars) == 0 else hold_out_covars

                transformer = Scaler(RobustScaler())
                train_transformed = transformer.fit_transform(train)
                my_model.fit(train_transformed, past_covariates=covariate)
                ts = transformer.transform(test)
                ts = train_transformed.concatenate(ts)

            # Perform backtesting on the hold-out participant
            retrain = False if not univariate else 5
            backtest_series = my_model.historical_forecasts(
                ts,
                past_covariates=covariate,
                start=val_cutoff,
                num_samples=num_samples,
                forecast_horizon=horizon,
                stride=1,
                last_points_only=False,
                retrain=retrain,
            )
            backtest_series = (
                concatenate(backtest_series, ignore_time_axis=True)
                if isinstance(backtest_series, list)
                else backtest_series
            )
            #transformed = transformer.inverse_transform(backtest_series)
            #model.save()
            if loss.__name__ == "mql":
                
                lossName = "MQLoss"

                q_80_low = backtest_series.quantile_df(0.2).rename(columns={"gl_0.2":f"{model.__class__.__name__}-lo-80"})
                q_80_high = backtest_series.quantile_df(0.8).rename(columns={"gl_0.8":f"{model.__class__.__name__}-hi-80"})
                q_90_low = backtest_series.quantile_df(0.1).rename(columns={"gl_0.1":f"{model.__class__.__name__}-lo-90"})
                q_90_high = backtest_series.quantile_df(0.9).rename(columns={"gl_0.9": f"{model.__class__.__name__}-hi-90"})
                q_median = backtest_series.quantile_df(0.5).rename(columns={"gl_0.5": f"{model.__class__.__name__}-median"})
                merged = pd.concat([q_80_low,q_80_high,q_90_low,q_90_high,q_median], axis=1).reset_index(drop=True)
                dataframe = hold_out_participant.loadData(prefix.replace("_ml", ""), df, horizon, lossName)
                dataframe = dataframe.reset_index()
                dataframe = dataframe.drop(columns=merged.columns.intersection(dataframe.columns), errors='ignore')
                dataframe = pd.concat([dataframe, merged],axis=1)
                #print(dataframe)
                hold_out_participant.saveData(prefix.replace("_ml", ""), df,dataframe, horizon, lossName)
               # prefix += f"_mql{level}"
            else:
                forecast_series = backtest_series.pd_dataframe()
                forecast_df = forecast_series.reset_index()
                forecast_df = forecast_df.rename(columns={"gl" : model.__class__.__name__})
                lossName = "RMSE"
                dataframe = hold_out_participant.loadData(prefix.replace("_ml", ""), df, horizon, lossName)
                dataframe = dataframe.drop(columns=forecast_df.columns.intersection(dataframe.columns), errors='ignore')
                #dataframe = dataframe.reset_index(drop=True)
                dataframe[f"{model.__class__.__name__}".replace("Model", "")] = backtest_series.values()
                hold_out_participant.saveData(prefix.replace("_ml", ""), df,dataframe, horizon, lossName)
                

            hold_out_participant.saveData(f"{prefix}_{model.__class__.__name__}", df, backtest_series, horizon, loss)
            hold_out_participant.saveData(f"{prefix}_{model.__class__.__name__}_model", df, model, horizon, loss)
            print(f"Cross learning completed for {hold_out_participant.Id} for model {model.__class__.__name__}")

# We expect a pkl file from running cross_learning first which we use for the cutoff value when training
def main():
    # Argument parser for hold-out key and other configurations
    global non_lipids
    parser = argparse.ArgumentParser(description="Cross learning with hold-out participant")
    parser.add_argument('--hold-out-key', type=str, required=True, help="Hold-out participant key")
    parser.add_argument("--run-name", "--rn", default="cross_learning_validated_lipids_metabolites", type=str)
    parser.add_argument("--error-index", "--ei", default=1, type=int)
    parser.add_argument("--horizon", "--h", default=2, type=int)
    parser.add_argument("--dimension", default=None, type=int)
    parser.add_argument("--univariate", default=False, type=bool)
    args = parser.parse_args()
    idx = args.error_index

    error_metrics = [mape,rmse,mql]
    horizon = args.horizon
    error = error_metrics[idx]
    run_name = args.run_name 

    script_dir = os.path.dirname(os.path.abspath(__file__))
    hold_out_key = args.hold_out_key  # Use the key provided via argparse
    if ("lipids" in run_name or "metabolites" in run_name) and hold_out_key in non_lipids:
        print("Skipping participant since participant does not have lipids data")
        return
    keys = get_directory_names(script_dir)
    loss = error_metrics[idx]
    cross_learn(run_name, keys, hold_out_key, loss, horizon, args.dimension, univariate=args.univariate)


if __name__ == "__main__":
    main()
