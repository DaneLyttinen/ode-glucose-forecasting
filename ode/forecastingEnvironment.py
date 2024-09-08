from pandas import DataFrame
class ForecastingEnvironment:
    def __init__(self, dataset: DataFrame, train_set_size, key):
        self.dataset = dataset
        self.train_set_size = train_set_size
        self.key = key
        self.states = []

    def prepare_initial_data(self):
        raise NotImplementedError

    def update_data(self, current_index):
        raise NotImplementedError
