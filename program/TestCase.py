# list the experiment case for OOBN
# packages
from DataLoader import make_dataloader
from OOBN import ObjectNode
from BN import Variable
from DataClalss import pt_data_types, walk_data_types, car_data_types

# example test case
def exTest():
    # Name of the case
    case_name = "example2"
    # Define the case
    data_files = [
        'data/activityData/MS2611_utf8.csv',
        'data/losData/05_代表徒歩_現況.csv',
        # 'data/losData/03_代表自動車_現況_utf8.csv',
        'data/losData/01_代表鉄道_現況_utf8.csv',
    ]
    convert_dict = {
        "自動車運転免許保有の状況": {0: 0, 1: 1, 2: 1},
        "代表交通手段：分類０": {1: 0, 5: 1},
        "移動の目的": {
            1: 0, 2: 0, 3: 1, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2, 19: 3
        },
        "目的種類：分類１": {
            2: 0, 3: 0, 4: 0, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 13: 2, 14: 1, 15: 0, 16: 0, 17: 1, 18: 1
        },
    }

    convert_dict_continuous = {
        "徒歩所要時間（分）": 3,
        #"時間（分）": 3,
        "鉄道乗車時間〔分〕": 3,
        "トリップ数": 3,
        "年齢": 3,
        "同行人数：人数": 3,
    }

    change_name_dict = {
        # "自動車運転免許保有の状況": "License",
        "トリップ数": "TripNumber",
        "代表交通手段：分類０": "Mode",
        "移動の目的": "Purpose",
        "目的種類：分類１": "PurposeType",
        "年齢": "Age",
        "同行人数：人数": "Companion",
        "徒歩所要時間（分）": "WalkTime",
        "鉄道乗車時間〔分〕": "TrainTime",
        # "時間（分）": "CarTime",
    }

    # Define the columns for ObjectNode 1 and 2
    col_obj1 = ["Age", "Companion", "Purpose", "PurposeType"]
    col_obj2 = ["TripNumber", "WalkTime", "TrainTime", "Mode"]

    # Load data using the new make_dataloader function
    print("Loading data...")
    dl = make_dataloader(data_files, convert_dict, convert_dict_continuous, change_name_dict, case_name)
    print("data num : ", len(dl.pt_data))

    # Create ObjectNode instances
    print("Making ObjectNodes...")
    obj1 = ObjectNode("obj1", {})
    obj1.set_data_from_dataloader(dl, col_obj1)
    
    obj2 = ObjectNode("Mode", {})
    obj2.set_data_from_dataloader(dl, col_obj2)

    # Set input/output data for obj2
    obj2.set_data(obj2.variables["TripNumber"].get_data(), 'input')
    obj2.set_data(obj2.variables["Mode"].get_data(), 'output')

    # Set obj2 as a variable in obj1
    obj1.add_variable(obj2)

    print("Estimation starts...")
    fixed_positions_obj1 = {"Age": 0, "Purpose": 1}
    obj1.structure_optimization(fixed_positions_obj1)

    fixed_positions_obj2 = {"TripNumber": 0, "Mode": 4}
    obj2.structure_optimization(fixed_positions_obj2)

    # Display the optimized structure
    print("Visualizing structure for obj1...")
    obj1.visualize_structure()
    
    print("Visualizing structure for obj2...")
    obj2.visualize_structure()

if __name__ == "__main__":
    exTest()