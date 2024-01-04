import sys

sys.setrecursionlimit(3000)

from model.BN import Variable
from model.OOBN import ObjectNode

# extension from Object Oriented Bayesian Network
# modify structure learning process
# so that it uses data of person who conducted the activity
# (like people who has done i-th activity for i-th activity object)
class DynamicNode(ObjectNode):
    def __init__(self, name, variables={}):
        super().__init__(name)
        self.name = name
        self.variables = variables
        self.inputs = []
        self.outputs = []
        self.input_data = self.data
        self.output_data = self.data
        self.input_states = self.states
        self.output_states = self.states
        self.ordering = []
        self.calc_time = 0
        self.score = 0
        self.trip_num = int(self.name[-1])
        self.manage_variable_name = None

    def set_use_row(self, numOfTrips, var=None):
        # i-th of activity is obtained from the name of the object
        self.use_row = row_of_ith_activity(numOfTrips, self.trip_num)
        for variable in self.variables:
            variable.use_row = self.use_row
        
        self.manage_variable_name = var

    # generate variables, based on cpt
    # generate for person who conducted the i-th activity
    def generate(self, num_samples, start_node=None):
        manage_variable = self.object_node.find_variable(self.manage_variable_name)
        manage_data = manage_variable.get_data('output')
        conducted_persons = (manage_data >= self.trip_num + 1)

        update_ordering = self.ordering.copy()
        if start_node is not None:
            update_ordering = update_ordering[update_ordering.index(start_node)+1:]
        for var_name in update_ordering:
            self.variables[var_name].generate(num_samples)
            # row of the person who didn't conduct the i-th activity is set to 0
            self.variables[var_name].data[~conducted_persons] = 0

        return None


# a function for specifying which row of DynamicNode is used for learning
def row_of_ith_activity(numOfTrips, i):
    return numOfTrips >= i