import sys

sys.setrecursionlimit(3000)

from model.BN import Variable
from model.OOBN import ObjectNode

# extension from Object Oriented Bayesian Network
# modify structure learning process
# so that it uses data of person who conducted the activity
# (like people who has done i-th activity for i-th activity object)
class DynamicNode(ObjectNode):
    def __init__(self, name, states, object_class, object_name, object_states):
        super().__init__(name, states=None) 
        self.name = name
        self.variables = variables
        self.input = []
        self.output = name
        self.input_data = self.data
        self.output_data = self.data
        self.input_states = self.states
        self.output_states = self.states
        self.ordering = []
        self.calc_time = 0
        self.score = 0

    def set_use_row(self, numOfTrips):
        # i-th of activity is obtained from the name of the object
        i = int(self.name[-1])
        self.use_row = row_of_ith_activity(numOfTrips, i)

# a function for specifying which row of DynamicNode is used for learning
def row_of_ith_activity(numOfTrips, i):
    return numOfTrips >= i