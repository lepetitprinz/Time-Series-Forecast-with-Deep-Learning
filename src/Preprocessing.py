import config
from WindowGenerator import WindowGenerator

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Preprocessing(object):
    COL_TARGET = 'amt'
    COL_DROP_SELL = ['pd_cd']
    COL_DATETIME = 'dt'

    def __init__(self):
        self.prod_group = {'가': 'g1', '나': 'g1', '다': 'g2', '라': 'g2', '마': 'g3', '바': 'g3'}
        self.scaler = None
        self.val_size = 0.2
        self.test_size = 0.1
        self.train_mean = 0
        self.train_std = 0

        self.data_prep = None

    def preprocess(self):
        print("Implement data preprocessing")

        # Load dataset
        data = pd.read_csv(config.INPUT_PATH, delimiter='\t', thousands=',')

        # Preprocess dataset
        data_prep = self.prep_data(data=data)

        # Convert target data type to float
        data_prep[self.__class__.COL_TARGET] = data_prep[self.__class__.COL_TARGET].astype(float)

        # Aggregate data by datetime
        data_agg_sum = data_prep.groupby(by=['dt']).sum()[['sales', 'amt']].reset_index()
        data_agg_avg = data_prep.groupby(by=['dt']).mean()['dc'].reset_index()
        data_agg = data_agg_sum.merge(data_agg_avg, on='dt')
        data_agg = data_agg.set_index(keys=self.__class__.COL_DATETIME)

        self.data_prep = data_agg
        # Handle outliers
        # if config.SMOOTH_YN:
        #     x = self.handle_outlier(data=x)

        # split data
        train_df, val_df, test_df = self.split_data(data=data_agg)

        # Scaling
        train_df, val_df, test_df = self.normalize(train_df=train_df, val_df=val_df, test_df=test_df)

        return train_df, val_df, test_df

    def prep_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # convert column names to lowercase
        data.columns = [col.lower() for col in data.columns]

        # add product group column
        group_func = np.vectorize(self.group_product)
        data['prod_group'] = group_func(data['pd_nm'].to_numpy())

        # drop unnecessary columns
        data = data.drop(columns=self.__class__.COL_DROP_SELL)

        # convert date feature to datetime type
        data[self.__class__.COL_DATETIME] = pd.to_datetime(data[self.__class__.COL_DATETIME], format='%Y%m%d')

        return data

    @staticmethod
    def handle_outlier(data: pd.DataFrame) -> pd.DataFrame:
        for i, col in enumerate(data.columns):
            min_val = 0
            max_val = 0
            if config.SMOOTH_METHOD == 'quantile':
                min_val = data[col].quantile(config.SMOOTH_RATE)
                max_val = data[col].quantile(1 - config.SMOOTH_RATE)

            elif config.SMOOTH_METHOD == 'sigma':
                mean = np.mean(data[col].values)
                std = np.std(data[col].values)
                min_val = mean - 2 * std
                max_val = mean + 2 * std

            data[col] = np.where(data[col].values < min_val, min_val, data[col].values)
            data[col] = np.where(data[col].values > max_val, max_val, data[col].values)

        return data

    def split_data(self, data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        n = len(data)
        train_df = data[: int(n * (1 - self.val_size - self.test_size))]
        val_df = data[int(n * (1 - self.val_size - self.test_size)): int(n * (1 - self.test_size))]
        test_df = data[int(n * (1 - self.test_size)):]

        return train_df, val_df, test_df

    # Normalize dataset
    def normalize(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        train_mean = train_df.mean()
        train_std = train_df.std()

        self.train_mean = train_mean
        self.train_std = train_std

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        return train_df, val_df, test_df

    # Group mapping function
    def group_product(self, prod):
        return self.prod_group[prod]
