import numpy as np
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
        model.variables[var["name"]].generate_random_cpt()

    return model

def generate_random_cpt(variable):
    num_states = [parent.get_states('output') for parent in variable.parents] + [variable.get_states('input')]
    cpt = np.random.rand(*num_states)
    cpt /= cpt.sum(axis=-1, keepdims=True)
    return cpt

