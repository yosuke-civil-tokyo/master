# config file to manage test case
# global test case
# dictionary of test cases
Configs = {
    "example2": {
        "case_name": "example2",
        "data_files": [
            'data/activityData/MS2611_utf8.csv',
            'data/losData/05_代表徒歩_現況.csv',
            # 'data/losData/03_代表自動車_現況_utf8.csv',
            'data/losData/01_代表鉄道_現況_utf8.csv'
        ],
        "convert_dict": {
            "自動車運転免許保有の状況": {0: 1, 1: 2, 2: 2},
            "代表交通手段：分類０": {1: 1, 5: 2},
            "移動の目的": {
                1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 19: 4
            },
            "目的種類：分類１": {
                2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 1, 8: 1, 9: 1, 10: 2, 11: 2, 13: 3, 14: 2, 15: 1, 16: 1, 17: 2, 18: 2
            },
        },
        "convert_dict_continuous": {
            "徒歩所要時間（分）": 3,
            #"時間（分）": 3,
            "鉄道乗車時間〔分〕": 3,
            "トリップ数": 3,
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
            # "時間（分）": "CarTime",
        },
        "objects": [
            {"name": "obj1",
              "variables": ["Age", "Companion", "Purpose", "PurposeType"],
              "fix": {"Age": 0, "Purpose": 1},
              "objs": ["Mode"]
            },
            {"name": "Mode",
              "variables": ["TripNumber", "WalkTime", "TrainTime", "Mode"],
              "input": ["TripNumber"],
              "output": ["Mode"],
              "fix": {"TripNumber": 0, "Mode": 4}
            }
        ],
        "numrows": 10000,
    },
    "example3": {
        "case_name": "example3",
        "data_files": [
            'data/activityData/MS2611_utf8.csv',
            'data/losData/05_代表徒歩_現況.csv',
            'data/losData/03_代表自動車_現況_utf8.csv',
            'data/losData/01_代表鉄道_現況_utf8.csv'
        ],
        "convert_dict": {
            "自動車運転免許保有の状況": {0: 1, 1: 2, 2: 2},
            "代表交通手段：分類０": {1: 1, 5: 2, 10: 3},
            "移動の目的": {
                1: 1, 2: 1, 3: 2, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3, 9: 3, 10: 3, 19: 4
            },
            "目的種類：分類１": {
                2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 1, 8: 1, 9: 1, 10: 2, 11: 2, 13: 3, 14: 2, 15: 1, 16: 1, 17: 2, 18: 2
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
            {"name": "obj1",
              "variables": ["Age", "Companion", "Purpose", "PurposeType"],
              "fix": {"Age": 0, "Purpose": 1},
              "objs": ["Mode"]
            },
            {"name": "Mode",
              "variables": ["TripNumber", "WalkTime", "TrainTime", "CarTime", "License", "Mode"],
              "input": ["TripNumber", "WalkTime"],
              "output": ["Mode"],
              "fix": {"Mode": 5}
            }
        ],
        "numrows": 10000,
    }
}
