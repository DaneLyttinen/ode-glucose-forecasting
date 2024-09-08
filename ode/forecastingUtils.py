from collections import namedtuple
from T2DMSimulator.patient.t2dpatient import T2DPatient, Action
recAction = namedtuple('ctrller_action', ['basal', 'bolus', 'meal','metformin', 'physical', 'time', 'times'])
from T2DMSimulator.glucose.GlucoseParameters import GlucoseParameters
import pandas as pd
import numpy as np

class ForecastingUtils:
    @staticmethod
    def simulate_glucose(patient_data, start, stop, t2dsimPatient: T2DPatient, canOnlySeeCurrentTime=False):
        state_hist = []
        for index in range(start, stop):
            cho_index = index if not canOnlySeeCurrentTime else start
            CHO = patient_data.loc[cho_index, "meal"]
            first = True
            for i in range(15):
                if first and CHO != 0:
                    print(f"eating cho {CHO}")

                patient_mdl_act = Action(
                    insulin_fast=0, CHO=CHO, insulin_long=0, metformin=0,
                    vildagliptin=0, physical=80., stress=0
                )
                placeholder = recAction(basal=0, meal=0, bolus=0, metformin=0, physical=0, time=30, times=[])
                CHO = 0
                t2dsimPatient.step(patient_mdl_act, placeholder)
                state_hist.append(t2dsimPatient.state)
                first = False
        return state_hist

    @staticmethod
    def apply_values(params: GlucoseParameters, param_values: list):
        glucose_params = [
            ('QGH', 0), ('QGG', 1), ('QGK', 2), ('QGP', 3),
            ('VIPF', 4), ('QIH', 5), ('TIP', 6),
            ('fg', 7), ('Kq1', 8), ('kabs', 9),
            ('c5', 10), ('d5', 11), ('c4', 12),
            ('ml0', 13), ('gammapan', 14), ('Sfactor', 15),
            ('Kout', 16), ('CF2', 17), ('VPHI', 18)
        ]
        for attr, idx in glucose_params:
            setattr(params.glucoseSubmodel, attr, param_values[idx])

    @staticmethod
    def create_states_df(states: np.array, relevant_indexes):
        state_history = {}
        for index in relevant_indexes:
            states_series = states[:, 0, index]
            state_history[f"state{index}"] = states_series
        return pd.DataFrame(state_history)
