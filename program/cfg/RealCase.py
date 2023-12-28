# config file to manage real-data case
Configs = {
    "real1": {
        "case_name": "real1",
        "data_files": [
            'data/activityData/MS2611_utf8.csv',
            'data/losData/05_代表徒歩_現況.csv',
            'data/losData/03_代表自動車_現況_utf8.csv',
            'data/losData/01_代表鉄道_現況_utf8.csv'
        ],
        "convert_dict": {
            "自動車運転免許保有の状況": {0: 1, 1: 2, 2: 2},
            "代表交通手段：分類０": {1: 1, 5: 2, 10: 3},
            "トリップ数": {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
        },
        "convert_dict_continuous": {
            "鉄道乗車時間〔分〕": 5,
            "年齢": 2,
        },
        "change_name_dict": {
            "自動車運転免許保有の状況": "License",
            "トリップ数": "TripNumber",
            "代表交通手段：分類０": "Mode",
            "年齢": "Age",
            "鉄道乗車時間〔分〕": "TrainTime",
        },
        "onetime_variables": ["TripNumber", "License", "Age"],
        "trip_variables": ["Mode", "TrainTime"],
        "maximum_number_of_trips": 5,
        "nan_delete_columns": ["TripNumber"],
        "objects": [
            {"name": "obj1",
              "variables": ["TripNumber", "Age", "License"],
              "fix": {},
              "objs": ["Trip"]
            },
            {"name": "Trip",
              "variables": ["Mode", "TrainTime"],
              "fix": {},
            },
        ],
    }
}