import time
import pandas as pd
import numpy as np

from BN import Variable
from OOBN import ObjectNode

# Class for Personal Features
class PersonalFeature:
    def __init__(self, features):
        self.features = features  # Dictionary to store various features

# Class for Schedule Type
class ScheduleType:
    def __init__(self, type, num_of_trips):
        self.type = type
        self.num_of_trips = num_of_trips

# Class for Schedule Data
class ScheduleData:
    def __init__(self, personal_feature, schedule_type):
        self.personal_feature = personal_feature
        self.schedule_type = schedule_type
        self.trip_data_list = []  # List of TripData objects

# Class for Trip Features
class TripFeature:
    def __init__(self, features):
        self.features = features  # Dictionary to store various features

# Class for Activity Features
class ActivityFeature:
    def __init__(self, features):
        self.features = features  # Dictionary to store various features

# Class for Trip Data
class TripData:
    def __init__(self, trip_feature, activity_feature):
        self.trip_feature = trip_feature
        self.activity_feature = activity_feature
        self.los = None  # LOS object
        self.origin_zone = None  # Zone object
        self.dest_zone = None  # Zone object

# Class for LOS data
class LOS:
    def __init__(self, features):
        self.features = features  # Dictionary to store various features

# Class for Zone data
class Zone:
    def __init__(self, features):
        self.features = features  # Dictionary to store various features

# DataLoader class with reduced memory usage and support for converting to Variable class
class DataLoader:
    def __init__(self):
        self.pt_data = None  # Dataframe to store PT data
        self.los_data = None  # Dataframe to store LOS data
        self.zone_data = None  # Dataframe to store Zone data

    def load_pt_data(self, pt_csv_file_path):
        # Read PT data from CSV with shift-jis encoding
        self.pt_data = pd.read_csv(pt_csv_file_path)
        
    def load_los_data(self, los_csv_file_path, column_names={}):
        # Read LOS data from CSV
        new_los_data = pd.read_csv(los_csv_file_path)
        pt_data = self.pt_data.dropna(subset=['出発地：ゾーンコード', '到着地：ゾーンコード'])[['出発地：ゾーンコード', '到着地：ゾーンコード']].astype(int)
        new_los_data[['発ゾーン', '着ゾーン']] = new_los_data[['発ゾーン', '着ゾーン']].astype(int)
        
        # Merge the new LOS data with PT data
        merged_data = pd.merge(pt_data, new_los_data, how='left',
                                left_on=['出発地：ゾーンコード', '到着地：ゾーンコード'],
                                right_on=['発ゾーン', '着ゾーン'])
        
        # Add the new column to existing LOS data if exists
        for key, value in column_names.items():
            if self.los_data is None:
                self.los_data = merged_data[['発ゾーン', '着ゾーン']+[key]].rename(columns={key: value})
            else:
                # concatenate the new column to existing LOS data
                self.los_data = pd.concat([self.los_data, merged_data[[key]].rename(columns={key: value})], axis=1)

    def load_zone_data(self, zone_csv_file_path):
        # Read Zone data from CSV
        self.zone_data = pd.read_csv(zone_csv_file_path)

    # extract columns, and remove rows with NaN
    def extract_data(self, df, column_list):
        return df[column_list].dropna()
        
    def to_variable(self, column_list):
        variables = {}
        for column_name in column_list:
            if column_name in self.pt_data.columns:
                data_array = self.pt_data[column_name].to_numpy()
                variable = Variable(column_name, states=int(np.max(data_array) + 1))
                variable.set_data(data_array)
                variables[column_name] = variable
            else:
                print(f"Warning: Column {column_name} not found in data.")
        return variables

    def get_data(self, column_list):
        return self.to_variable(column_list)
    
    # make a table combining pt_data and los_data with specified columns
    def make_los_table(self, column_list):
        # concatenate pt_data and los_data
        table = pd.concat([self.pt_data, self.los_data], axis=1)
        # extract specified columns
        table = table[column_list]
        return table
    
    # discretize columns
    def discretize_dataframe(self, df, col_dict):
        for col_name, bin_size in col_dict.items():
            df[col_name] = pd.qcut(df[col_name].rank(method='first'), bin_size, labels=False)
        return df
    

# let's make simple los data for each trip
def make_walk_car(
        use_col=["自動車運転免許保有の状況", "WalkTime", "CarTime", "トリップ番号", "代表交通手段：分類０"], 
        change_name={"自動車運転免許保有の状況": "License", "WalkTime": "WalkTime", "CarTime": "CarTime", "トリップ番号": "TripNumber", "代表交通手段：分類０": "Mode"}):
    dl = DataLoader()
    dl.load_pt_data('data/activityData/MS2611_utf8.csv')
    dl.load_los_data('data/losData/05_代表徒歩_現況.csv', {'徒歩所要時間（分）': 'WalkTime'})
    dl.load_los_data('data/losData/03_代表自動車_現況_utf8.csv', {'時間（分）': 'CarTime'})

    table = dl.make_los_table(use_col)
    table = table[(table["代表交通手段：分類０"] == 1)|(table["代表交通手段：分類０"] == 10)]

    table = dl.extract_data(table, use_col)
    table = dl.discretize_dataframe(table, {"自動車運転免許保有の状況": 3, "WalkTime": 3, "CarTime": 3, "トリップ番号": 3, "代表交通手段：分類０": 2})
    table = table.rename(columns=change_name)

    # set the table as dl.pt_data, with reset index
    dl.pt_data = table.reset_index(drop=True)

    return dl


# Test code (you can replace 'pt_data.csv' with your actual PT data CSV file)
if __name__ == '__main__':
    dl = make_walk_car()