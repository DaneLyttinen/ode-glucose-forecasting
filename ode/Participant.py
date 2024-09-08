import os
import pickle
import pandas as pd
import re
from datetime import datetime
import os

participantPattern = re.compile(r'\b[a-zA-Z]+[0-9]{5}\b')
class Participant():
    def __init__(self, Id, treatment: str, saveFormat='pickle'):
        self.Id = Id
        self.path = os.path.join(os.path.dirname(os.path.abspath(__file__)), str(self.Id))
        self.treatment = treatment
        self.dataframes = self.__getDataframes()
        self.saveFormat = saveFormat
        self.__load_participant_static_vars()
        
    # Change as needed, important part is having the mean glucose available somehow for a dataset
    def __load_participant_static_vars(self):
        if not (participantPattern.search(str(self.Id))):
            return
        path = os.path.join(os.path.dirname(self.path),"participant_vars.xlsx")
        data_sheet = pd.read_excel(path, sheet_name='Data')

        result_dict = {}
        for index, row in data_sheet.iterrows():
            if (str(self.Id) in row["Study_ID"]):
                cid = row['CID']
                row_dict = row.drop(labels='CID').to_dict()
                if pd.notna(row['Gluc']):
                    result_dict[cid] = row_dict
        self.participant_static_vars = result_dict
    
    def get_participant_static_var(self, key, dataframe):
        study_start_time = self.dataframes[0].loc[0, "time"]
        #dataframe.name
        unchanged_dataframe = [df for df in self.dataframes if df.loc[0, "time"] == dataframe.loc[0, "time"]][0]
        study_start_time = datetime.strptime(study_start_time, '%Y-%m-%d %H:%M:%S') if isinstance(study_start_time, str) else study_start_time
        curr_df_start_time = datetime.strptime(unchanged_dataframe.loc[0, "time"], '%Y-%m-%d %H:%M:%S')  if unchanged_dataframe["time"].apply(lambda x: isinstance(x, str)).all() else unchanged_dataframe.loc[0, "time"]

        difference_in_days = abs((study_start_time - curr_df_start_time).days)
        index = 1
        if difference_in_days > 30.44:
            index = 4
        return self.participant_static_vars[index][key]


    def getDataFrames(self):
        return self.dataframes

    def __getDataframes(self):
        dfs =[]
        if (participantPattern.search(str(self.Id))):
            dfs = self.__getparticipantParticipantDataframes()
        return dfs
    
    def __getparticipantParticipantDataframes(self):
        if (self.treatment.find('augment') != -1):
            directory = self.path
            fileNames = [f.replace(".csv", "") for f in os.listdir(directory) if f.endswith('.csv') and not f.startswith(str(self.Id))]
            return self.__readDataframes(fileNames)
        filenames = self.__readAllCsvFilesStartingWithId()
        if ("multivariate" not in self.treatment):
            filenames = [file for file in filenames if "multivariate" not in file]
        else:
            filenames = [file for file in filenames if "multivariate" in file]
            if ("acc_rand" in self.treatment):
                filenames = [file for file in filenames if "acc_rand" in file]
            elif ("acc" in self.treatment):
                filenames = [file for file in filenames if "acc_rand" not in file and "acc" in file]
            else:
                filenames = [file for file in filenames if "acc" not in file]
        return self.__readDataframes(filenames)
    
    def __readAllCsvFilesStartingWithId(self):
        directory = self.path
        fileNames = [f.replace(".csv", "") for f in os.listdir(directory) if f.endswith('.csv') and f.startswith(str(self.Id))]
        return fileNames

    def __readDataframes(self, fileNames):
        dfs = []
        for fileName in fileNames:
            baselinePath = os.path.join(self.path, fileName+".csv")
            baseline = pd.read_csv(baselinePath)
            baseline.name = fileName
            dfs.append(baseline)
        return dfs
    
    def checkDataExists(self, prefix, dataset, data, horizon, loss):
        print(f"checking path {os.path.join(self.path,f'{prefix}_{self.Id}_{self.treatment}_{dataset.name}_{horizon}_{loss.__class__.__name__ }.pkl')}")
        return os.path.exists(os.path.join(self.path,f'{prefix}_{self.Id}_{self.treatment}_{dataset.name}_{horizon}_{loss.__class__.__name__ }.pkl'))

    def saveData(self, prefix, dataset, data, horizon, loss):
        #prefix = "result" if result else "all_forecasts"
        with open(os.path.join(self.path,f'{prefix}_{self.Id}_{self.treatment}_{dataset.name}_{horizon}_{loss.__class__.__name__ }.pkl'), 'wb') as f:
            pickle.dump(data, f)

    def loadData(self, prefix, dataset, horizon, loss):
        with open(os.path.join(self.path, f'{prefix}_{self.Id}_{self.treatment}_{dataset.name}_{horizon}_{loss.__class__.__name__}.pkl'), 'rb') as f:
            data = pickle.load(f)
        return data