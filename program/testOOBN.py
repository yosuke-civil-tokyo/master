from OOBN import ObjectNode
from BN import Variable 
from DataLoader import DataLoader, make_walk_car

def main():
    use_col={"自動車運転免許保有の状況": 3, "WalkTime": 3, "CarTime": 3, "トリップ番号": 3, "代表交通手段：分類０": 2,
             "移動の目的":3, "目的種類：分類１":3, "年齢":3, "同行人数：人数":3}
    change_name= {"自動車運転免許保有の状況": "License", "トリップ番号": "TripNumber", "代表交通手段：分類０": "Mode",
                  "移動の目的": "Purpose", "目的種類：分類１": "PurposeType", "年齢": "Age", "同行人数：人数": "Companion"}
    new_use_col = [change_name.get(item, item) for item in use_col.keys()]
    col_obj1 = ["Age", "Companion", "Purpose", "PurposeType"]
    col_obj2 = ["License", "TripNumber", "WalkTime", "CarTime", "Mode"]

    # load data
    print("Loading data...")
    dl = make_walk_car(use_col, change_name)
    # let's make a ObjectNode
    print("Making ObjectNode...")
    obj1 = ObjectNode("obj1", {})
    obj1.set_data_from_dataloader(dl, col_obj1)
    obj2 = ObjectNode("ModeSelect", {})
    obj2.set_data_from_dataloader(dl, col_obj2)

    # set input/output data of obj2
    obj2.set_data(obj2.variables["License"].get_data(), 'input')
    obj2.set_data(obj2.variables["Mode"].get_data(), 'output')

    # set obj2 as a variable in obj1
    obj1.add_variable(obj2)

    print("Estimation starts...")
    fixed_positions = {"Age": 0, "Purpose":1}
    obj1.structure_optimization(fixed_positions)

    fixed_positions = {"License": 0, "Mode":4}
    obj2.structure_optimization(fixed_positions)

    # Display the optimized structure
    print("Visualizing structure...")
    obj1.visualize_structure()
    obj2.visualize_structure()


if __name__ == "__main__":
    main()