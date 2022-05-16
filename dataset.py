from datetime import timedelta

import pandas as pd
from os.path import exists
import util

class Dataset:

    @staticmethod
    def getDataDir():
        return './_data/refit/'
    def __init__(self):

        self.data_dir=Dataset.getDataDir()
        self.processed_dir=Dataset.getDataDir()
    def getCSVPath(self,homeNum,processed):
        path = self.data_dir + f'CLEAN_House{homeNum}.csv'
        if processed:
            path = self.processed_dir + f'House_{homeNum}.csv'
        return path
    def getPicklePath(self,homeNum,processed):
        path = self.data_dir + f'CLEAN_House{homeNum}.pkl'
        if processed:
            path = self.processed_dir + f'House_{homeNum}.pkl'
        return path

    def getHouseDataCSV(self, homeNum:int, processed=True):
        path=self.getCSVPath(homeNum,processed)
        return pd.read_csv(path)

    def getHouseDataFrame(self, houseNum:int, processed=True)->pd.DataFrame:

        path=self.getPicklePath(houseNum,processed)
        if not exists(path):
            self.to_pickleHome(houseNum,processed)
        df= pd.read_pickle(path)
        df.columns = util.house[houseNum]
        df.index.names = ['datetime']
        return df

    def to_pickleHome(self,houseNum,processed=True):
        df=self.getHouseDataCSV(houseNum, processed)
        df['Unix'] = pd.to_datetime(df['Unix'], unit='s')
        df.set_index('Unix',inplace=True)
        del df['Time']
        picklePath = self.getPicklePath(houseNum, processed)

        df.to_pickle(picklePath)
    def getHouseDataByDate(self,house=1,first=0,days=1):
        df = Dataset().getHouseDataFrame(house)
        start = df.index[0].date() + timedelta(days=first)
        end = start + timedelta(days=(first + days))
        return df[start:end]

