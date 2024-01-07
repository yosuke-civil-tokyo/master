import sys
import pandas as pd
import numpy as np

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

    # generate data, based on cpt
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
    
    # generate function, for specified person
    def generate_for_part(self, num_samples, return_rows):
        manage_variable = self.object_node.find_variable(self.manage_variable_name)
        manage_data = manage_variable.get_data('output')
        conducted_persons = (manage_data >= self.trip_num + 1)

        for var_name in self.ordering:
            resampled_data = self.variables[var_name].generate(num_samples, set_data=False)
            # row of the person who didn't conduct the i-th activity is set to 0
            self.variables[var_name].data[~conducted_persons] = 0
            # change row of the person who resampled activity
            self.variables[var_name].data[return_rows] = resampled_data[return_rows]

# a function for specifying which row of DynamicNode is used for learning
def row_of_ith_activity(numOfTrips, i):
    return numOfTrips >= i

# a function for generating data
# generate data, and resample it considering the time constraints
# now just consider one constraint
# 1. time constraints between activities, replacing activities that overlaps or comes before the previous activity
def generate_data_with_constraint(num_samples, target_obj=ObjectNode("exp"), start_node=None):
    # generate data
    target_obj.generate(num_samples, start_node)
    whole_data = target_obj.make_table()
    original_data = whole_data.copy()
    whole_variables = whole_data.columns
    max_trip_num = max([int(col[4]) for col in whole_variables if (col.startswith("Trip") and col[4].isdigit())])

    # resample data
    resampled_data, resampled_rows = resample_with_constraint(target_obj, whole_data, max_trip_num)
    return original_data, resampled_data, resampled_rows

# a function for resampling data
# resample data, considering time constraints
def resample_with_constraint(target_obj=ObjectNode("exp"), whole_data=pd.DataFrame(), max_trip_num=5):
    iteration = 0
    resampled_rows = np.zeros(len(whole_data), dtype=bool)
    j = 0
    while (iteration < 100) & (j < max_trip_num-1):
        # resampling process
        # get the activity that doesn't satisfy the time constraints
        j = 0
        for i in range(1, max_trip_num):
            # calculate the finished time of i-th activity
            # note that start_time's unit is 4-hours, and activity_time's unit is 2-hours
            start_time = (whole_data[f"Trip{i}_StartTime"].values - 1) * 4
            stay_time = (whole_data[f"Trip{i}_ActivityTime"].values - 1) * 2
            end_time = start_time + stay_time
            # get the start time of (i+1)-th activity
            next_start_time = (whole_data[f"Trip{i+1}_StartTime"].values - 1) * 4

            rows_of_constraint = (end_time+4 > next_start_time)
            resampled_rows_num = rows_of_constraint.sum()
            resampled_rows = resampled_rows | rows_of_constraint

            if resampled_rows_num == 0:
                j += 1
                continue
            
            # resample the activity
            target_obj.find_variable(f"Trip{i}").generate_for_part(len(whole_data), rows_of_constraint)
            whole_data = target_obj.make_table()
        iteration += 1
        print(f"{iteration}th iteration, {resampled_rows.sum()} rows are resampled")

    return whole_data, resampled_rows