from model.OOBN import ObjectNode
from model.BN import Variable 
from data.DataLoader import DataLoader, make_walk_car

def main():
    use_col={"自動車運転免許保有の状況": 3, "WalkTime": 3, "CarTime": 3, "トリップ数": 3, "代表交通手段：分類０": 2,
             "移動の目的":3, "目的種類：分類１":3, "年齢":3, "同行人数：人数":3}
    change_name= {"自動車運転免許保有の状況": "License", "トリップ数": "TripNumber", "代表交通手段：分類０": "Mode",
                  "移動の目的": "Purpose", "目的種類：分類１": "PurposeType", "年齢": "Age", "同行人数：人数": "Companion"}
    new_use_col = [change_name.get(item, item) for item in use_col.keys()]

    # load data
    print("Loading data...")
    dl = make_walk_car(use_col, change_name)
    # let's make a ObjectNode
    print("Making ObjectNode...")
    object_node = ObjectNode("TestNode", {})
    object_node.set_data_from_dataloader(dl, new_use_col)

    print("Estimation starts...")
    fixed_positions = {"Mode": 8}
    object_node.structure_optimization(fixed_positions)

    # Display the optimized structure
    print("Visualizing structure...")
    object_node.visualize_structure()


if __name__ == "__main__":
    main()
