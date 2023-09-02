# main script for Object Oriented Bayesian Network

import numpy as np
import pandas as pd

# class for object
class Object:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.variables = data.columns.values
        self.input_var = []
        self.output_var = []
        self.child_object = []
        self.parent_object = []

    # set input variables
    def set_input(self, input_var):
        self.input_var.append(input_var)

    # set output variables
    def set_output(self, output_var):
        self.output_var.append(output_var)

    # set children objects
    def set_child(self, child_object):
        self.child_object.append(child_object)
        child_object.parent_object.append(self)

    # set parent objects
    def set_parent(self, parent_object):
        self.parent_object.append(parent_object)
        parent_object.child_object.append(self)

    # set a function to compute output_var from input_var
    def set_function(self, function):
        self.function = function

    # calculate output_var from input_var
    def forward(self):
        return self.function(self.input_var)
    



# Example usage
# function to create random output regardless of input
def random_function(input_var):
    return np.random.normal()

# let's test it
obj1 = Object('Object1', ['A', 'B', 'C'])
obj1.set_input(['A', 'B'])
obj1.set_output(['C'])
obj1.set_function(random_function)

obj2 = Object('Object2', ['X', 'Y'])
obj2.set_input(['X'])
obj2.set_output(['Y'])
obj2.set_function(random_function)

obj1.set_child(obj2)

