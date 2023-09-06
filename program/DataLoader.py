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
        merged_data = pd.merge(pt_data, new_los_data, how='inner',
                                left_on=['出発地：ゾーンコード', '到着地：ゾーンコード'],
                                right_on=['発ゾーン', '着ゾーン'])
        print(new_los_data)
        print(pt_data)
        print(merged_data)
        
        # Add the new column to existing LOS data if exists
        for key, value in column_names.items():
            if self.los_data is None:
                self.los_data = merged_data[['発ゾーン', '着ゾーン']+[key]].rename(columns={key: value})
            else:
                self.los_data = pd.merge(self.los_data, merged_data[key], how='inner', on=['発ゾーン', '着ゾーン']).rename(columns={key: value})

    def load_zone_data(self, zone_csv_file_path):
        # Read Zone data from CSV
        self.zone_data = pd.read_csv(zone_csv_file_path)
        
    def to_variable(self, column_list):
        variables = {}
        for column_name in column_list:
            if column_name in self.pt_data.columns:
                data_array = self.pt_data[column_name].to_numpy()
                variable = Variable(column_name, states=np.max(data_array) + 1)
                variable.set_data(data_array)
                variables[column_name] = variable
            else:
                print(f"Warning: Column {column_name} not found in data.")
        return variables

    def get_data(self, column_list):
        return self.to_variable(column_list)

# Test code (you can replace 'pt_data.csv' with your actual PT data CSV file)
if __name__ == '__main__':
    dl = DataLoader()
    dl.load_pt_data('data/activityData/MS2611_utf8.csv')
    # dl.load_los_data('los_data.csv')
    # dl.load_zone_data('zone_data.csv')
    object_node = ObjectNode("TestNode", {})
    dl.load_los_data('data/losData/05_代表徒歩_現況.csv', {'徒歩所要時間（分）': 'WalkTime'})
    print(dl.los_data)
    dl.load_los_data('data/losData/02_代表バス_現況_utf8.csv', {'ゾーン間所要時間（分）': 'BusTime'})
    print(dl.los_data)  # This should print a dictionary of updated Variable objects for the specified columns
