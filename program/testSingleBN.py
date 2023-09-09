from OOBN import ObjectNode
from BN import Variable 
from DataLoader import DataLoader, make_walk_car

def main():
    use_col=["自動車運転免許保有の状況", "WalkTime", "CarTime", "トリップ番号", "代表交通手段：分類０"]

    # load data
    print("Loading data...")
    dl = make_walk_car(use_col)
    # let's make a ObjectNode
    print("Making ObjectNode...")
    object_node = ObjectNode("TestNode", {})
    object_node.set_data_from_dataloader(dl, use_col)

    print("Estimation starts...")
    object_node.structure_optimization()

    # Display the optimized structure
    print("Visualizing structure...")
    object_node.visualize_structure()

if __name__ == "__main__":
    main()
