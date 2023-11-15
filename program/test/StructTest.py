from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

# list the structure optimization experiment case
import argparse
# packages
from cfg.StructCase import Configs
from dl.DataLoader import make_dataloader
from model.OOBN import ObjectNode
from model.BN import Variable

# example test case
def exTest(config):
    case_name = config["case_name"]
    data_files = config["data_files"]
    convert_dict = config["convert_dict"]
    convert_dict_continuous = config["convert_dict_continuous"]
    change_name_dict = config["change_name_dict"]
    object_configs = config["objects"]
    evaluate_target = config["evaluate_target"]

    # Load data using the new make_dataloader function
    print("Loading data...")
    dl = make_dataloader(data_files, convert_dict, convert_dict_continuous, change_name_dict, case_name)
    print("data num : ", len(dl.pt_data))

    # use config["numrows"] to limit the number of rows
    dl.pt_data = dl.pt_data[:config["numrows"]]
    dl.train_test_split()
    dl.pt_data = dl.train_data
    
    # list of object
    objects = {}

    # Iterate through the object configurations and initialize ObjectNode instances
    for obj_conf in object_configs:
        print(f"Making {obj_conf['name']}...")
        obj_columns = [change_name_dict.get(var, var) for var in obj_conf["variables"]]  # Handle renamed columns
        objects[obj_conf['name']] = ObjectNode(obj_conf['name'], {})
        objects[obj_conf['name']].set_data_from_dataloader(dl, obj_columns)

        # If the object has defined input/output variables, set them
        if "input" in obj_conf and "output" in obj_conf:
            for input_var in obj_conf["input"]:
                input_var = change_name_dict.get(input_var, input_var)
                objects[obj_conf['name']].set_data(objects[obj_conf['name']].variables[input_var].get_data(), input_var, 'input')
            for output_var in obj_conf["output"]:
                output_var = change_name_dict.get(output_var, output_var)
                objects[obj_conf['name']].set_data(objects[obj_conf['name']].variables[output_var].get_data(), output_var, 'output')

    # Now, iterate again to set up object relationships after all have been initialized
    for obj_conf in object_configs:
        if "objs" in obj_conf:
            for child_name in obj_conf["objs"]:
                objects[obj_conf['name']].add_variable(objects[child_name])

    # Perform structure optimization based on fixed positions
    for obj_conf in object_configs:
        fixed_positions = {change_name_dict.get(k, k): v for k, v in obj_conf["fix"].items()}
        print(f"Structure optimization for {obj_conf['name']}...")
        objects[obj_conf['name']].structure_optimization(fixed_positions)

    # evaluate performance with test data
    # reset data with test
    dl.pt_data = dl.test_data
    for obj_conf in object_configs:
        print(f"Making {obj_conf['name']}...")
        obj_columns = [change_name_dict.get(var, var) for var in obj_conf["variables"]]  # Handle renamed columns
        objects[obj_conf['name']].set_data_from_dataloader(dl, obj_columns)

        # If the object has defined input/output variables, set them
        if "input" in obj_conf and "output" in obj_conf:
            for input_var in obj_conf["input"]:
                input_var = change_name_dict.get(input_var, input_var)
                objects[obj_conf['name']].set_data(objects[obj_conf['name']].variables[input_var].get_data(), input_var, 'input')
            for output_var in obj_conf["output"]:
                output_var = change_name_dict.get(output_var, output_var)
                objects[obj_conf['name']].set_data(objects[obj_conf['name']].variables[output_var].get_data(), output_var, 'output')
    
    # evaluate performance with log likelihood
    print("Evaluating performance...")
    for obj in objects:
        if evaluate_target in obj.variables:
            obj.evaluate(evaluate_target)


    # Visualize the structures
    for obj_conf in object_configs:
        print(f"Visualizing structure for {obj_conf['name']}...")
        try:
            objects[obj_conf['name']].visualize_structure()
        except:
            print("Error: Could not visualize structure of Object \"{}\"".format(obj_conf['name']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a structure optimization experiment")

    # add the arguments
    parser.add_argument("CaseName",
                        metavar="case_name",
                        type=str,
                        help="the name of the case to run")
    args = parser.parse_args()
    config = Configs[args.CaseName]

    exTest(config)