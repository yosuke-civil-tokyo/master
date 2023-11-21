from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import argparse
from cfg.TrueModel import Configs
from model.BN import Variable
from model.OOBN import ObjectNode

def BuildModelFromConfig(config):
    # read config
    variables = config["variables"]

    # build model
    model = ObjectNode("model", {})
    for var in variables:
        # new variable
        new_var = Variable(var["name"], var["states"])
        if "parents" in var:
            new_var.set_parents([model.variables[parent] for parent in var["parents"]])

        # add variable to model
        model.add_variable(new_var)

    # generate random cpt
    for var in variables:
        model.variables[var["name"]].set_random_cpt()

    model.ordering = [var["name"] for var in variables]

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build model from config file.')
    parser.add_argument("--ModelConfig", metavar="model_config", type=str, default="model1", help='config name')
    parser.add_argument("--NumSamples", metavar="num_samples", type=int, default=10000, help='number of samples')
    parser.add_argument("--OutputFile", metavar="output_file", type=str, default="model1", help='output file name')
    args = parser.parse_args()
    config = Configs[args.ModelConfig]

    model = BuildModelFromConfig(config)
    model.generate(args.NumSamples)
    model.save_data(args.OutputFile, "csv")