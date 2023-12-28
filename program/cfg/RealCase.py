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
            "代表交通手段：分類０": {1: 1, 2: 2, 5: 3, 6: 3, 7: 4, 8: 4, 10: 5},
            "トリップ数": {0: 1, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
            "性別": {1: 1, 2: 2},
            "自由に使える自動車の有無": {1: 1, 2: 1, 3: 2},
            "就業（形態・状況）": {1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2},
            "移動の目的": {
                1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 19: 4
            },
            "目的種類：分類３": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
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
            "性別": "Sex",
            "自由に使える自動車の有無": "CarAvailable",
            "就業（形態・状況）": "Employment",
            "移動の目的": "Purpose",
            "目的種類：分類３": "PurposeType",
        },
        "onetime_variables": ["TripNumber", "License", "Age", "Sex", "CarAvailable", "Employment"],
        "trip_variables": ["Mode", "TrainTime", "Purpose", "PurposeType"],
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