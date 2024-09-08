from ode.forecastingEnvironment import ForecastingEnvironment
from ode.forecastingUtils import ForecastingUtils
from T2DMSimulator.glucose.GlucoseParameters import GlucoseParameters
from T2DMSimulator.patient.t2dpatient import T2DPatient, Action
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import copy

class MultivariateForecastingEnvironment(ForecastingEnvironment):
    def __init__(self, dataset, train_set_size, key,participant, initial_params=None, optimize=False, plot=False, gen_future=False):
        super().__init__(dataset, train_set_size, key)
        self.optimize = optimize
        self.initial_params = initial_params
        self.participant = participant
        self.simulatorPatient = self._initialize_simulator(initial_params)
        self.meal_indexes = dataset[dataset['meal'] != 0].index.tolist()
        self.gen_future = gen_future

    def _initialize_simulator(self, initial_params):
        if initial_params == None:
            initial_params = [
                43.7, 10.1, 10.1, 15.1, 6.74, 3.12, 20.0, 0.9, 0.68, 0.08,
                2.03, 1.59, 1.72, 6.33, 2.366, 1, 68.30411374407583,
                21.151177251184837, 11.31
            ]
        glucoseParams = GlucoseParameters()
        #ForecastingUtils.apply_values(glucoseParams, initial_params)
        avg_gl = self.participant.get_participant_static_var("Gluc", self.dataset) * 18
        simulatorPatient = T2DPatient({}, glucose_params=glucoseParams, name=self.key, GBPC0=avg_gl)
        return simulatorPatient

    def prepare_initial_data(self):
        last_fit_index = 0
        initial_params = [
            43.7, 10.1, 10.1, 15.1, 6.74, 3.12, 20.0, 0.9, 0.68, 0.08,
            2.03, 1.59, 1.72, 6.33, 2.366, 1, 68.30411374407583,
            21.151177251184837, 11.31
        ] if self.initial_params == None else self.initial_params
        train_size = self.train_set_size if not self.gen_future else self.train_set_size + 8
        for i in range(train_size):
            current_index = i
            if self.optimize and i >= 5 and self.dataset.iloc[i]['meal'] > 0 and i - last_fit_index >= 5:
                fit_start = max(0, i - 5)
                fit_end = min(i + 5, self.train_set_size)
                optimized_params = self.fit_ode(initial_params, self.dataset, self.simulatorPatient, fit_start, fit_end)
                initial_params = optimized_params
                optimized_glucose_params = GlucoseParameters()
                #ForecastingUtils.apply_values(optimized_glucose_params, optimized_params)
                self.simulatorPatient.param = optimized_glucose_params
                last_fit_index = i
            # ensure it doesn't go stiff
            if (len(self.meal_indexes) > 0 and min(self.meal_indexes) > 200 and i % 200 == 180):
                self.simulatorPatient = self._initialize_simulator(None)
                state = ForecastingUtils.simulate_glucose(self.dataset, i-20, i + 1, self.simulatorPatient)
                state = state[::15]
                self.states.append([state[-1]])
                continue
            state = ForecastingUtils.simulate_glucose(self.dataset, current_index, current_index + 1, self.simulatorPatient, self.gen_future)
            self.states.append(state[::15])
            print(f"Processed data point {i+1}/{self.train_set_size}")

        state_history = ForecastingUtils.create_states_df(np.array(self.states), self._get_relevant_indexes())
        return pd.concat([self.dataset[:self.train_set_size], state_history], axis=1)

    def update_data(self, current_index):
        self._update_state(current_index)
        states = self.states[-8:len(self.states)] if self.gen_future else [self.states[-1]]
        new_state_df = ForecastingUtils.create_states_df(np.array(states), self._get_relevant_indexes())
        return new_state_df

    def _update_state(self, index):
        stop = index + 8 if self.gen_future else index + 1
        state = ForecastingUtils.simulate_glucose(self.dataset, index, stop, self.simulatorPatient,self.gen_future)
        self.states.append(state[::15])
        print(f"Processed data point {stop}/{len(self.dataset)}")

    @staticmethod
    def objective_function(values_and_data):
        patient_data, param_values, t2dsimPatient, start, stop = values_and_data
        params = GlucoseParameters()
        ForecastingUtils.apply_values(params, param_values)
        t2dsimPatient.param = params
        gl_list = patient_data["gl"].to_list()
        try:
            simulated_values = ForecastingUtils.simulate_glucose(patient_data, start, stop, t2dsimPatient)
            real_data = gl_list[start:stop]
            simulated_glucose = [array[34] for array in simulated_values[::15]]
            real_data = [round(value * 18, 1) for value in real_data]
            mse = np.mean((np.array(simulated_glucose) - np.array(real_data))**2)
            print(mse)
        except Exception as e:
            print(e)
            mse = 2000000
        return mse

    def create_objective_wrapper(self, dataset, t2dsimPatient, start, stop):
        def wrapped_objective(initial_params):
            patient_copy = copy.deepcopy(t2dsimPatient)
            return self.objective_function((dataset, initial_params, patient_copy, start, stop))
        return wrapped_objective

    def fit_ode(self, initial_params, dataset, t2dsimPatient: T2DPatient, start, stop):
        wrapped_objective = self.create_objective_wrapper(dataset, t2dsimPatient, start, stop)
        result = minimize(wrapped_objective, initial_params, method='L-BFGS-B')
        optimized_params = result.x
        optimized_glucose_params = GlucoseParameters()
        ForecastingUtils.apply_values(optimized_glucose_params, optimized_params)
        t2dsimPatient.param = optimized_glucose_params
        return optimized_params

    def _get_relevant_indexes(self):
        return [0, 1, 2, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45]