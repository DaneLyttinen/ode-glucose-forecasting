import numpy as np
import copy
from scipy.optimize import differential_evolution
from collections import namedtuple
from T2DMSimulator.patient.t2dpatient import T2DPatient, Action
from T2DMSimulator.glucose.GlucoseParameters import GlucoseParameters
recAction = namedtuple('ctrller_action', ['basal', 'bolus', 'meal','metformin', 'physical', 'time', 'times'])
import pandas as pd
import numpy as np
from ode.Participant import Participant
from datetime import datetime
import multiprocessing as mp
import pickle

def simulate_glucose(patient_data, start, stop, t2dsimPatient):
    state_hist = []
    for index in range(start, stop):
        CHO = patient_data.loc[index, "meal"]
        for _ in range(15):
            tmp_cho = 5 if CHO > 0 else 0
            patient_mdl_act = Action(
                insulin_fast=0, CHO=tmp_cho, insulin_long=0, metformin=0,
                vildagliptin=0, physical=80.0, stress=0
            )
            placeholder = recAction(basal=0, meal=0, bolus=0, metformin=0, physical=0, time=30, times=[])
            CHO = max(0, CHO - 5)
            t2dsimPatient.step(patient_mdl_act, placeholder)
            state_hist.append(t2dsimPatient.state)
    return state_hist

def compare_glucose_curves(simulated, actual):
    """
    Compare simulated and actual glucose curves using Pearson correlation and RMSE.
    """
    sim = np.array([array[34] for array in simulated[::15]])
    
    actual = np.array([x * 18 for x in actual])
    diff = actual[0] - sim[0]
    actual += abs(diff)
    simulated = []
    actuals = []
    for i,num in enumerate(sim):
        simulated.append([num,i])
        actuals.append([actual[i],i])
    rmse = np.sqrt(np.mean((actual - sim)**2))
    return rmse

def find_expanding_windows(indexes, threshold=10):
    if not indexes:
        return []

    windows = []
    current_window = [indexes[0]]

    for i in range(1, len(indexes)):
        if indexes[i] - indexes[i - 1] <= threshold:
            current_window.append(indexes[i])
        else:
            windows.append(current_window)
            current_window = [indexes[i]]
    
    windows.append(current_window)  # Append the last window

    return windows

def evaluate(cho_combination, df, start, stop, t2dsimPatient, meal_indexes):
    copied_patient = copy.deepcopy(t2dsimPatient)
    copied_df = df.copy(deep=True)
    
    # Convert scaled CHO values back to original scale
    cho_combination = [cho * 5 for cho in cho_combination]
    
    # Apply CHO values to respective meal indexes
    for meal_index, cho in zip(meal_indexes, cho_combination):
        copied_df.at[meal_index, 'meal'] = int(cho)
    
    simulated_values = simulate_glucose(copied_df, start, stop, copied_patient)
    rmse = compare_glucose_curves(simulated_values, df.loc[start:stop-1, 'gl'].to_list())
    
    return rmse

def find_best_cho_content(df, start, stop, t2dsimPatient, meal_indexes, max_cho=100):
    """
    Find the best matching CHO content for a given set of actual glucose values,
    considering multiple potential meal times within the start-stop window using differential evolution.
    """
    # Scale the bounds to reflect steps of 5
    bounds = [(0, max_cho // 5) for _ in meal_indexes]  # CHO values between 0 and max_cho in steps of 5
    
    result = differential_evolution(evaluate, bounds, args=(df, start, stop, t2dsimPatient, meal_indexes), 
                                    strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7,
                                    init='latinhypercube', polish=False, updating='deferred', workers=1)
    
    # Convert the results back to multiples of 5
    best_cho_combination = dict(zip(meal_indexes, [cho * 5 for cho in result.x]))
    
    print("Best CHO combination:")
    for index, cho in best_cho_combination.items():
        print(f"Index: {index}, CHO: {cho:.2f}g")
    
    return best_cho_combination

def estimate_unknown_cho_contents(t2dsimPatient, dataframe, indexes):
    """
    Estimate CHO contents for unknown meal data points based on baseline data.
    """
    
    # Initialize with baseline data
    all_cho_contents = []
    all_actual_glucose = []
    
    # Find CHO contents for unknown data points
    estimated_cho = []
    processed_window_max = 0
    windows = find_expanding_windows(indexes)
    for i, glucose_value in enumerate(dataframe['gl'].to_list()):
        all_actual_glucose.append(glucose_value)
        if i in indexes and i > processed_window_max:
            
            start = i
            
            idx_in_window = [x for x in windows if x[0] == i]
            if (len(idx_in_window) > 0):
                idx_in_window = idx_in_window[0]
            processed_window_max = idx_in_window[-1] if len(idx_in_window) > 0 else 0
            stop = processed_window_max + 11 if processed_window_max > 0 else (start + 11)
            stop = min(stop, len(dataframe['gl']-1))

            best_cho = find_best_cho_content(
                dataframe,
                start,
                stop,
                t2dsimPatient,
                idx_in_window,
            )
            estimated_cho.extend(best_cho.values())
            all_cho_contents.append(best_cho)
            print(estimated_cho)
    return estimated_cho
    
def remove_near_indexes(indexes):
    result = []
    for i in range(len(indexes)):
        if i == 0:
            result.append(indexes[i])
        elif indexes[i] - result[-1] >= 2:
            result.append(indexes[i])
    return result

def parse_array_to_datetimes(array):
    parsed_array = []
    time_to_meal_dict = {}
    for element in array:
        try:
            # Attempt to parse the element as a datetime
            temp = element
            if (";" in element):
                temp = element[:element.index(";")]
                meal = element[element.index(";")+1:]
                meal = int(meal)
                time_to_meal_dict[temp] = meal
                continue
            parsed_element = datetime.strptime(temp, '%d/%m/%Y %H:%M')
        except (ValueError, TypeError):
            # If parsing fails, keep the element in its original form
            parsed_element = element
        parsed_array.append(parsed_element)
    return parsed_array, time_to_meal_dict


def process_participant(item):
    participant_id, meal_dict = item
    participant = Participant(participant_id, "online")
    dataframes = participant.dataframes
    parsed_array, time_to_meal_dict = parse_array_to_datetimes(meal_dict["meal"])
    cho_cont_dict = {}
    
    for df in dataframes:
        try:
            avg_gl = participant.get_participant_static_var("Gluc", df) * 18
            simulatorPatient = T2DPatient({}, glucose_params=GlucoseParameters(), name=participant_id, GBPC0=avg_gl)
            cho_cont_dict[df.name] = {}
            df["meal"] = 0
            
            for time, meal in time_to_meal_dict.items():
                try:
                    parsed_time = datetime.strptime(time, '%d/%m/%Y %H:%M')
                    parsed_time = parsed_time.strftime("%Y-%m-%d %H:%M:%S")
                    if parsed_time in df["time"].values:
                        idx = df["time"].to_list().index(parsed_time)
                        df.at[idx, 'meal'] = meal
                except:
                    continue
            
            df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
            indexes = df.index[df["time"].isin(parsed_array)].tolist()
            filtered_indexes = remove_near_indexes(indexes)
            cho_cont_dict[df.name]["indexes"] = filtered_indexes
            estimated_cho_contents = estimate_unknown_cho_contents(simulatorPatient, df, filtered_indexes)
            cho_cont_dict[df.name]["meals"] = estimated_cho_contents
            
            print(f"cho contnents for participant {participant_id} {estimated_cho_contents}")
            participant.saveData("cho_data_fixed", df, cho_cont_dict)
        except Exception as e:
            print(f"error occured for participant {participant_id} on dataframe {df.name}")
            print(e)
            continue
    return participant_id, cho_cont_dict

def main():
    # Expected to look as below, 50 is CHO {'AA12345': {'meal': ['14/01/2023 09:00;50', '14/01/2023 11:00']}}
    participant_to_meals_dict = {}#
    results = {}
    for participant_id, meal_dict in participant_to_meals_dict.items():
        id, cho_cont_dict = process_participant((participant_id, meal_dict))
        results[participant_id] = cho_cont_dict

    
    # Combine results
    combined_cho_cont_dict = dict(results)
    print(combined_cho_cont_dict)
    
    with open('combined_cho_cont_dict.pkl', 'wb') as f:
        pickle.dump(combined_cho_cont_dict, f)

if __name__ == '__main__':
    main()