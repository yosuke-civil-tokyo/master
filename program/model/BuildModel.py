from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import argparse
from model.BN import Variable
from model.OOBN import ObjectNode
from model.DBN import DynamicNode

def BuildModelFromConfig(config):
    variables = {}
    # Create and add Variables to their respective ObjectNodes
    for var_name, var_info in config["variables"].items():
        new_var = Variable(var_name, var_info["num_states"])
        variables[var_name] = new_var
    
    for var_name, var_info in config["variables"].items():
        # Assign parents to the variable
        new_var = variables[var_name]
        if var_info.get("parents"):
            new_var.set_parents([variables[parent_name] for parent_name in var_info["parents"]])

        # Generate random CPT for the variable
        if var_info.get("cpt"):
            new_var.set_cpt(var_info["cpt"])
        else:
            new_var.set_random_cpt()

    # Create and add ObjectNodes to the model
    objects = {}
    for obj_name, obj_info in config["objects"].items():
        if obj_info.get("dynamic", False):
            new_obj = DynamicNode(obj_name, {})
        else:
            new_obj = ObjectNode(obj_name, {})
        for var_name in obj_info["variables"]:
            new_obj.add_variable(variables[var_name])
        new_obj.ordering = obj_info["variables"]
        objects[obj_name] = new_obj

    for obj_name, obj_info in config["objects"].items():
        if obj_info.get("in_obj"):
            for child_name in obj_info["in_obj"]:
                objects[obj_name].add_variable(objects[child_name])

    return objects["obj1"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build model from config file.')
    parser.add_argument("--ModelConfig", metavar="model_config", type=str, default="model1", help='config name')
    parser.add_argument("--NumSamples", metavar="num_samples", type=int, default=10000, help='number of samples')
    parser.add_argument("--OutputFile", metavar="output_file", type=str, default="model1", help='output file name')
    args = parser.parse_args()
    with open(args.ModelConfig, "r") as f:
        config = json.load(f)

    model = BuildModelFromConfig(config)
    model.generate(args.NumSamples)
    model.save_data(args.OutputFile, "csv")
    model.save_model_parameters(args.OutputFile+"/truth")