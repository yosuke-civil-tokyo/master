# config file to manage test case
# test cases for testing structure optimization methods' performance
# dictionary of test cases
Configs = {
    "struct1": {
        "case_name": "struct1",
        "data_files": [
            'data/activityData/MS2611_utf8.csv',
            'data/losData/05_代表徒歩_現況.csv',
            'data/losData/03_代表自動車_現況_utf8.csv',
            'data/losData/01_代表鉄道_現況_utf8.csv'
        ],
        "convert_dict": {
            "自動車運転免許保有の状況": {1: 0, 2: 1, 3: 1},
            "代表交通手段：分類０": {1: 0, 5: 1, 10: 2},
            "移動の目的": {
                1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 19: 3
            },
            "目的種類：分類１": {
                2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 13: 2, 14: 1, 15: 0, 16: 0, 17: 1, 18: 1
            },
        },
        "convert_dict_continuous": {
            "徒歩所要時間（分）": 3,
            "時間（分）": 5,
            "鉄道乗車時間〔分〕": 5,
            "トリップ数": 10,
            "年齢": 3,
            "同行人数：人数": 3,
        },
        "change_name_dict": {
            "自動車運転免許保有の状況": "License",
            "トリップ数": "TripNumber",
            "代表交通手段：分類０": "Mode",
            "移動の目的": "Purpose",
            "目的種類：分類１": "PurposeType",
            "年齢": "Age",
            "同行人数：人数": "Companion",
            "徒歩所要時間（分）": "WalkTime",
            "鉄道乗車時間〔分〕": "TrainTime",
            "時間（分）": "CarTime",
        },
        "objects": [
            {"name": "whole",
              "variables": ["Age", "Companion", "Purpose", "PurposeType", "TripNumber", "WalkTime", "TrainTime", "CarTime", "License", "Mode"],
              "fix": {},
            }
        ],
    }
}