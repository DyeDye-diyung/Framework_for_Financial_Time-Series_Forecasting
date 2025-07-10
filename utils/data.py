from torch.utils.data import Dataset
import pandas as pd


class StockDataSet(Dataset):

    def __init__(self, df: pd.DataFrame, target='Apple', is_scaled=True):
        self.df = df
        self.length, self.dim = self.df.shape
        self.is_scaled = is_scaled  # Whether to apply scaling

        target_column = df.pop('Close')  # Extract the target column
        df['Close'] = target_column  # Put the target column back at the end of the df

        self.X = df.values  # Original data
        self.Y = df['Close'].values.reshape(-1, 1)  # Reshape 1D array to 2D

        if self.is_scaled:
            # Scaling
            # from sklearn.preprocessing import StandardScaler
            from sklearn.preprocessing import MinMaxScaler
            self.x_scaler = MinMaxScaler()
            self.y_scaler = MinMaxScaler()
            self.X = self.x_scaler.fit_transform(self.X)
            self.Y = self.y_scaler.fit_transform(self.Y)
        else:
            # Do not perform scaling
            self.x_scaler = None
            self.y_scaler = None

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.length

    def inverse_transform(self, y):
        if self.y_scaler is not None:
            return self.y_scaler.inverse_transform(y)
        else:
            # raise ValueError("inverse_transform not available since data was not scaled.")
            return y

    @classmethod
    def from_preprocessed(cls, path='data/processed_dataset.csv', target='Apple', is_scaled=True):
        return cls(pd.read_csv(path, index_col=0, parse_dates=True).astype('float32'), target, is_scaled)

    @classmethod
    def from_raw(cls, folder_path='data', target='Apple', is_scaled=True):
        import os
        from .preprocess import merge, preprocess
        # df = merge({name[:-16]: pd.read_csv(f'{folder_path}/{name}', index_col=0, parse_dates=True)
        #             for name in os.listdir(folder_path) if name.endswith('(2017-2023).csv')})
        df = merge({name[:-16]: pd.read_csv(f'{folder_path}/{name}', index_col=0, parse_dates=True)
                    for name in os.listdir(folder_path) if name.endswith('(2015-2024).csv')})
        df = preprocess(df, target).dropna().astype('float32')
        return cls(df, target, is_scaled)
