import os
import time
import pandas as pd
import numpy as np

from model.BN import Variable
from model.OOBN import ObjectNode

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

    def load_pt_data(self, pt_csv_file_path, data_types=None):
        # Read PT data from CSV with shift-jis encoding
        self.pt_data = pd.read_csv(pt_csv_file_path)
        
    def load_los_data(self, los_csv_file_path, data_types=None):
        # Read LOS data from CSV
        new_los_data = pd.read_csv(los_csv_file_path)
        pt_data = self.pt_data.dropna(subset=['出発地：ゾーンコード', '到着地：ゾーンコード'])[['出発地：ゾーンコード', '到着地：ゾーンコード']].astype(int)
        new_los_data[['発ゾーン', '着ゾーン']] = new_los_data[['発ゾーン', '着ゾーン']].astype(int)
        
        # Merge the new LOS data with PT data
        merged_data = pd.merge(pt_data, new_los_data, how='left',
                                left_on=['出発地：ゾーンコード', '到着地：ゾーンコード'],
                                right_on=['発ゾーン', '着ゾーン'])
        
        # Add the new column to existing LOS data if exists
        if self.los_data is None:
            self.los_data = merged_data
        else:
            # concatenate the new column to existing LOS data
            self.los_data = pd.concat([self.los_data, merged_data], axis=1)

    def load_zone_data(self, zone_csv_file_path):
        # Read Zone data from CSV
        self.zone_data = pd.read_csv(zone_csv_file_path)

    # extract columns, and remove rows with NaN
    def extract_data(self, df, column_list):
        return df[column_list].dropna()
    
    # use rows with NaN by replacing with 0
    def fill_data(self, df, column_list):
        return df[column_list].fillna(0)
        
    def to_variable(self, column_list, dataRange=None):
        if dataRange is None:
            dataRange = (0, len(self.pt_data))
        variables = {}
        for column_name in column_list:
            if column_name in self.pt_data.columns:
                data_array = self.pt_data[column_name].iloc[dataRange[0]:dataRange[1]].to_numpy()
                variable = Variable(column_name, states=int(np.max(data_array) + 1))
                variable.set_data(data_array)
                variables[column_name] = variable
            else:
                print(f"Warning: Column {column_name} not found in data.")
        return variables

    def get_data(self, column_list, dataRange=None):
        return self.to_variable(column_list, dataRange)
    
    # make a table combining pt_data and los_data
    def make_los_table(self):
        # concatenate pt_data and los_data
        table = pd.concat([self.pt_data, self.los_data], axis=1)
        return table
    
    def discretize_dataframe(self, df, col_dict):
        for col_name, value_mapping in col_dict.items():
            if col_name in df.columns:
                # Create a dictionary to map values using the provided conversion mapping
                # conversion_dict = {int(key): value_mapping.get(int(key), np.nan) for key in df[col_name]}
                df[col_name] = df[col_name].astype(int).map(value_mapping)
        
        # Remove rows with NaN
        df = df.fillna(0)
        return df

    # by continuous value
    def discretize_dataframe_fromcon(self, df, col_dict):
        for col_name, bin_size in col_dict.items():
            print(col_name)
            df[col_name] = pd.qcut(df[col_name].astype(float).rank(method='first'), bin_size, labels=False).values + 1
        return df
    
    # split data for train and test
    def train_test_split(self, split_ratio=0.8):
        # Split the data into train and test sets
        total_data = self.pt_data
        split_index = int(len(total_data) * split_ratio)
        train_data = total_data[:split_index]
        test_data = total_data[split_index:]

        # set data
        self.train_data = train_data
        self.test_data = test_data
        return train_data, test_data
    

# let's make simple los data for each trip
def make_walk_car(
        use_col={"自動車運転免許保有の状況": 3, "WalkTime": 3, "CarTime": 3, "トリップ番号": 3, "代表交通手段：分類０": 2}, 
        change_name={"自動車運転免許保有の状況": "License", "WalkTime": "WalkTime", "CarTime": "CarTime", "トリップ番号": "TripNumber", "代表交通手段：分類０": "Mode"},
        return_table=False):
    dl = DataLoader()
    dl.load_pt_data('data/activityData/MS2611_utf8.csv')
    dl.load_los_data('data/losData/05_代表徒歩_現況.csv', {'徒歩所要時間（分）': 'WalkTime'})
    dl.load_los_data('data/losData/03_代表自動車_現況_utf8.csv', {'時間（分）': 'CarTime'})

    table = dl.make_los_table(use_col.keys())
    table = table[(table["代表交通手段：分類０"] == 1)|(table["代表交通手段：分類０"] == 10)]

    table = dl.extract_data(table, use_col.keys())
    table = dl.discretize_dataframe_fromcon(table, use_col)
    table = table.rename(columns=change_name)

    # set the table as dl.pt_data, with reset index
    dl.pt_data = table.reset_index(drop=True)

    if return_table:
        return table

    return dl

def make_dataloader(
        data_files=None, 
        convert_dict=None,
        convert_dict_continuous=None,
        change_name_dict=None,
        case_name=None, 
        return_table=False):
    dl = DataLoader()

    # Check if intermediate data exists
    intermediate_file_path = f'./data/midData/{case_name}.csv'
    if case_name and os.path.exists(intermediate_file_path):
        print(f"Loading intermediate data from {intermediate_file_path}")
        dl = DataLoader()
        dl.pt_data = pd.read_csv(intermediate_file_path, dtype=int)
        return dl
    
    # Load data files if provided
    print("Start loading data")
    if data_files is not None:
        for file_path in data_files:
            if "activityData" in file_path:
                print("read activity data: {}".format(file_path))
                dl.load_pt_data(file_path)
            elif "losData" in file_path:
                print("read los data: {}".format(file_path))
                dl.load_los_data(file_path)
            elif "zoneData" in file_path:
                print("read zone data: {}".format(file_path))
                dl.load_zone_data(file_path)

    print("los processing")
    table = dl.make_los_table()

    print("descretization")
    table = dl.fill_data(table, list(convert_dict.keys())+list(convert_dict_continuous.keys()))
    table = dl.discretize_dataframe(table, convert_dict)
    table = dl.discretize_dataframe_fromcon(table, convert_dict_continuous)
    table = table.rename(columns=change_name_dict).astype(int)

    # Set the table as dl.pt_data, with reset index
    dl.pt_data = table.reset_index(drop=True)

    print("Table:")
    print(dl.pt_data)

    # Save intermediate data to a CSV file
    if case_name:
        print(f"Saving intermediate data to {intermediate_file_path}")
        os.makedirs('./data/midData', exist_ok=True)
        dl.pt_data.to_csv(intermediate_file_path, index=False)

    if return_table:
        return table

    return dl


# let's make pt_data only table
def make_trip(use_col={"移動の目的":3, "目的種類：分類１":3, "年齢":3, "同行人数：人数":3}):
    return None


# Test code (you can replace 'pt_data.csv' with your actual PT data CSV file)
if __name__ == '__main__':
    dl = make_walk_car()