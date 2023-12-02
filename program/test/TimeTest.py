import os
from pathlib import Path
import sys
import time
sys.path.append(str(Path(__file__).resolve().parent.parent))

# list the structure optimization experiment case
import argparse
# packages
from cfg.StructCase import Configs
from dl.DataLoader import make_dataloader
from model.OOBN import ObjectNode
from model.BN import Variable

# example test case
def exTest(config, CaseName=""):
    case_name = config.get("case_name")
    data_files = config.get("data_files")
    convert_dict = config.get("convert_dict")
    convert_dict_continuous = config.get("convert_dict_continuous")
    change_name_dict = config.get("change_name_dict")
    object_configs = config["objects"]
    evaluate_target = config["evaluate_target"]
    structure_opt = config["structure_opt"]

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
        obj_columns = [var for var in obj_conf["variables"]]  # Handle renamed columns
        objects[obj_conf['name']] = ObjectNode(obj_conf['name'], {})
        objects[obj_conf['name']].set_data_from_dataloader(dl, obj_columns)

        # If the object has defined input/output variables, set them
        if "input" in obj_conf and "output" in obj_conf:
            for input_var in obj_conf["input"]:
                objects[obj_conf['name']].set_data(objects[obj_conf['name']].variables[input_var].get_data(), input_var, 'input')
            for output_var in obj_conf["output"]:
                objects[obj_conf['name']].set_data(objects[obj_conf['name']].variables[output_var].get_data(), output_var, 'output')

    # Now, iterate again to set up object relationships after all have been initialized
    for obj_conf in object_configs:
        if "objs" in obj_conf:
            for child_name in obj_conf["objs"]:
                objects[obj_conf['name']].add_variable(objects[child_name])

    # Perform structure optimization
    flag = 0
    start_time = time.time()
    duration = config.get("duration", 0)
    while time.time() - start_time < duration:
        for obj_conf in object_configs:
            fixed_positions = {k: v for k, v in obj_conf["fix"].items()}
            print(f"Structure optimization for {obj_conf['name']}...")
            if structure_opt == "order_optimization":
                objects[obj_conf['name']].order_optimization(fixed_positions)
            elif structure_opt == "greedy_structure_learning":
                objects[obj_conf['name']].greedy_structure_learning()
            elif structure_opt == "tabu_structure_learning":
                objects[obj_conf['name']].tabu_structure_learning()
            else:
                raise ValueError("Invalid structure optimization method!")
            
        # evaluate performance with test data
        # reset data with test
        if config.get("evaluate", False):
            dl.pt_data = dl.test_data
            for obj_conf in object_configs:
                print(f"Making {obj_conf['name']}...")
                obj_columns = [var for var in obj_conf["variables"]]  # Handle renamed columns
                objects[obj_conf['name']].set_data_from_dataloader(dl, obj_columns)

                # If the object has defined input/output variables, set them
                if "input" in obj_conf and "output" in obj_conf:
                    for input_var in obj_conf["input"]:
                        objects[obj_conf['name']].set_data(objects[obj_conf['name']].variables[input_var].get_data(), input_var, 'input')
                    for output_var in obj_conf["output"]:
                        objects[obj_conf['name']].set_data(objects[obj_conf['name']].variables[output_var].get_data(), output_var, 'output')

            # evaluate performance with log likelihood
            print("Evaluating performance...")
            for obj in objects.values():
                if evaluate_target in obj.variables:
                    obj.evaluate(evaluate_target)

        # Visualize the structures
        if config.get("visualize", False):
            print(f"Visualizing structure")
            try:
                objects.get("obj1").visualize_structure()
            except:
                print("Error: Could not visualize structure of Object \"{}\"".format(obj_conf['name']))


        # save the model
        if not os.path.exists(os.path.join("data", "modelData", case_name, CaseName+structure_opt)):
            os.makedirs(os.path.join("data", "modelData", case_name, CaseName+structure_opt))
        objects.get("obj1").save_model_parameters(os.path.join(case_name, CaseName+structure_opt, str(flag)))
        flag += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a structure optimization experiment")

    # add the arguments
    parser.add_argument("CaseName",
                        metavar="case_name",
                        type=str,
                        help="the name of the case to run")
    parser.add_argument("StructureOpt",
                        metavar="structure_opt",
                        type=str,
                        help="the name of the structure optimization method to run")
    args = parser.parse_args()
    config = Configs[args.CaseName]
    config["structure_opt"] = args.StructureOpt

    exTest(config, CaseName=args.CaseName)