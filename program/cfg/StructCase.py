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
            {"name": "whole",
              "variables": ["Age", "Companion", "Purpose", "PurposeType", "TripNumber", "WalkTime", "TrainTime", "CarTime", "License", "Mode"],
              "fix": {},
            }
        ],
        "numrows": 1000,
        "evaluate_target": "Mode",
    },
    "fast1": {
        "case_name": "fast1",
        "data_files": [
            'data/activityData/MS2611_utf8.csv',
            'data/losData/05_代表徒歩_現況.csv',
            'data/losData/03_代表自動車_現況_utf8.csv',
            'data/losData/01_代表鉄道_現況_utf8.csv'
        ],
        "convert_dict": {
            "自動車運転免許保有の状況": {0: 1, 1: 2, 2: 2},
            "代表交通手段：分類０": {1: 1, 5: 2, 10: 3},
        },
        "convert_dict_continuous": {
            "鉄道乗車時間〔分〕": 5,
            "トリップ数": 10,
            "同行人数：人数": 3,
        },
        "change_name_dict": {
            "自動車運転免許保有の状況": "License",
            "トリップ数": "TripNumber",
            "代表交通手段：分類０": "Mode",
            "同行人数：人数": "Companion",
            "鉄道乗車時間〔分〕": "TrainTime",
        },
        "objects": [
            {"name": "obj1",
              "variables": ["TrainTime", "License"],
              "fix": {},
              "objs": ["Trip"]
            },
            {"name": "Trip",
              "variables": ["Companion", "TripNumber", "Mode"],
              "input": ["Companion", "TripNumber"],
              "output": ["Mode"],
              "fix": {0: "TripNumber", 1: "Companion"},
            }
        ],
        "numrows": 10000,
        "evaluate_target": "Mode",
        "visualize": True,
    },
    "model1": {
        "case_name": "model1",
        "objects": [
            {"name": "obj1",
              "variables": ["access", "license", "ageGroup", "employed", "trip1", "trip2", "act2", "act3", "trip3", "act4", "trip4", "act5", "trip5", "act6", "trip6", "act7"],
              "fix": {},
              "objs": [],
            }
        ],
        "numrows": 50000,
        "evaluate_target": "trip1",
        "flags": 30,
        "duration": 100,
    },
    "model2-obj": {
        "case_name": "model2",
        "objects": [
            {"name": "obj1",
              "variables": ["age", "gender", "education", "license", "access", "income", "employed", "Totaltripduration", "Totalactivityduration"],
              "fix": {13: "Totaltripduration", 14: "Totalactivityduration"},
              "objs": ["t1", "t2", "t3", "t4", "t5", "t6"],
            },
            {"name": "t1",
             "variables": ["trip1", "tDur1", "act2", "aDur2"],
             "input": ["trip1", "act2"],
             "output": ["act2"],
             "fix": {},
             "objs": [],
            },
            {"name": "t2",
             "variables": ["trip2", "tDur2", "act3", "aDur3"],
             "input": ["trip2"],
             "output": ["act3"],
             "fix": {},
             "objs": [],
            },
            {"name": "t3",
             "variables": ["trip3", "tDur3", "act4", "aDur4"],
             "input": ["trip3"],
             "output": ["act4"],
             "fix": {},
             "objs": [],
            },
            {"name": "t4",
             "variables": ["trip4", "tDur4", "act5", "aDur5"],
             "input": ["trip4"],
             "output": ["act5"],
             "fix": {},
             "objs": [],
            },
            {"name": "t5",
             "variables": ["trip5", "tDur5", "act6", "aDur6"],
             "input": ["trip5"],
             "output": ["act6"],
             "fix": {},
             "objs": [],
            },
            {"name": "t6",
             "variables": ["trip6", "tDur6", "act7", "aDur7"],
             "input": ["trip6"],
             "output": ["act7"],
             "fix": {},
             "objs": [],
            },
        ],
        "numrows": 50000,
        "evaluate_target": "trip1",
        "flags": 1,
        "duration": 3600,
        "visualize": False,
    },
    "model2-normal": {
        "case_name": "model2",
        "objects": [
            {"name": "obj1",
              "variables": ["age", "gender", "education", "license", "access", "income", "employed", "trip1", "tDur1", "act2", "aDur2", "trip2", "tDur2", "act3", "aDur3", "trip3", "tDur3", "Totaltripduration", "act4", "aDur4", "trip4", "tDur4", "Totalactivityduration", "act5", "aDur5", "trip5", "tDur5", "act6", "aDur6", "trip6", "tDur6", "act7", "aDur7"],
              "fix": {},
              "objs": [],
            },
        ],
        "numrows": 50000,
        "evaluate_target": "trip1",
        "flags": 1,
        "duration": 3600,
        "visualize": False,
    },
}