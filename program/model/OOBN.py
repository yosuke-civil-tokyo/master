import os
import math
import json
from itertools import chain, combinations
import numpy as np
import pandas as pd
import random
import time
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import sys

sys.setrecursionlimit(3000)

from model.BN import Variable


# ObjectNode class definition
class ObjectNode(Variable):
    def __init__(self, name, variables={}):
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
    
    def set_data(self, data_array, variable_name=None, data_type='input'):
        if data_type == 'input':
            self.input_data = np.array(data_array)
            self.input_states=int(np.max(data_array) + 1)
            self.input.append(variable_name)
        elif data_type == 'output':
            self.output_data = np.array(data_array)
            self.output_states=int(np.max(data_array) + 1)
            self.output = variable_name

    def get_variables(self, data_type='input'):
        if data_type == 'input':
            return {var_name: self.variables[var_name] for var_name in self.input}
        
    def find_variable(self, var_name):
        # Check if the variable is directly in this object
        if var_name in self.variables:
            return self.variables[var_name]

        # If not, search in child objects
        for child in self.variables.values():
            if isinstance(child, ObjectNode):
                result = child.find_variable(var_name)
                if result is not None:
                    return result

        return None

    def get_data(self, data_type='input'):
        if data_type == 'input':
            return self.input_data
        elif data_type == 'output':
            return self.output_data
        else:
            raise ValueError("Invalid data_type. Choose 'input' or 'output'.")

    def get_states(self, data_type='input'):
        if data_type == 'input':
            return self.input_states
        elif data_type == 'output':
            return self.output_states
        else:
            raise ValueError("Invalid data_type. Choose 'input' or 'output'.")
    
    def add_variable(self, variable):
        self.variables[variable.name] = variable
        variable.object_node = self

    def order_optimization(self, fixed_positions=None):
        startTime = time.time()
        # check if any variable has to be fixed
        if fixed_positions is None:
            fixed_positions = {}
    
        initial_ordering = [k for k in self.variables.keys() if k not in fixed_positions.values()]
        random.shuffle(initial_ordering)

        # Apply fixed positions if provided
        current_ordering = [None for _ in range(len(self.variables))]
        for i in range(len(current_ordering)):
            if i in fixed_positions.keys():
                current_ordering[i] = fixed_positions[i]

            else:
                current_ordering[i] = initial_ordering.pop(0)

        best_score = float('-inf')
        improvement = True
        swap_pairs = [[i, i+1] for i in range(len(current_ordering) - 1) if not {i, i+1} & set(fixed_positions.keys())]
        
        # search for the best ordering, until no improvement is found by swapping
        with ProcessPoolExecutor() as executor:
            while improvement:
                improvement = False
                results = executor.map(self.evaluate_swap, swap_pairs, repeat(current_ordering))
                for score, ordering in results:
                    if score > best_score:
                        best_score = score
                        current_ordering = ordering
                        improvement = True

        self.ordering = current_ordering
        self.update_structure(self.ordering)
        final_score = self.BIC_all()
        print("Final Score : ", final_score)
        self.calc_time = time.time() - startTime
        self.score = final_score

    def evaluate_swap(self, pair, ordering):
        ordering = ordering.copy()
        ordering[pair[0]], ordering[pair[1]] = ordering[pair[1]], ordering[pair[0]]
        # update structure
        score = self.score_ordering(ordering)
        # score = self.BIC_all()
        return score, ordering

    def score_ordering(self, ordering):
        name_to_cpt = {}
        name_to_parents = {}

        for var_name in ordering:
            if var_name in self.input:
                continue

            preceding_vars = ordering[:ordering.index(var_name)]
            variable = self.variables[var_name]
            # if the variable is an object node, iterate over its input variables
            if isinstance(variable, ObjectNode):
                for input_var_name in variable.input:
                    input_variable = variable.variables[input_var_name]
                    # assuming that the input variable is not an object node
                    best_parents, best_cpt = self.find_optimal_parents(input_variable, preceding_vars)
                    name_to_parents[input_var_name] = best_parents
                    name_to_cpt[input_var_name] = best_cpt
            else:
                best_parents, best_cpt = self.find_optimal_parents(variable, preceding_vars)
                name_to_parents[var_name] = best_parents
                name_to_cpt[var_name] = best_cpt

        total_score = sum(self.find_variable(var_name).temp_BIC_score([self.find_variable(parent_name) for parent_name in name_to_parents[var_name]], name_to_cpt[var_name]) for var_name in name_to_parents.keys())
        return total_score

    def BIC_all(self):
        score = 0
        
        for variable in self.variables.values():
            if isinstance(variable, ObjectNode):
                # calculate the score for each input variable in the object node
                for input_var_name in variable.input:
                    input_variable = variable.variables[input_var_name]
                    score += input_variable.BIC_sep()
            else:
                score += variable.BIC_sep()
        
        return score
    
    def ll_all(self):
        score = 0
        
        for variable in self.variables.values():
            if isinstance(variable, ObjectNode):
                # calculate the score for each input variable in the object node
                for input_var_name in variable.input:
                    input_variable = variable.variables[input_var_name]
                    score += input_variable.ll_sep()
            else:
                score += variable.ll_sep()
        
        return score
    
    def calculate_LL0(self, variable):
        # For a variable with k states, the log-likelihood under the null model is N * log(1/k)
        k = variable.get_states('input')
        N = len(variable.get_data('input'))
        log_likelihood = N * math.log(1 / k)
        return log_likelihood

    def update_structure(self, ordering):
        for var_name in ordering:
            # skip if the variable is input type in this object node
            if var_name in self.input:
                continue

            variable = self.variables[var_name]
            # if the variable is an object node, iterate over its input variables
            if isinstance(variable, ObjectNode):
                preceding_vars = ordering[:ordering.index(var_name)]
                for input_var_name in variable.input:
                    input_variable = variable.variables[input_var_name]
                    # assuming that the input variable is not an object node
                    self.set_optimal_parents(input_variable, preceding_vars)

            else:
                preceding_vars = ordering[:ordering.index(var_name)]
                self.set_optimal_parents(variable, preceding_vars)

    def find_optimal_parents(self, variable, preceding_vars):
        # print("Finding optimal parents for variable: ", variable)
        best_parents = []
        best_cpt = None
        best_score = float('-inf')
        
        LL0 = self.calculate_LL0(variable)

        for r in range(min(len(preceding_vars) + 1, 3)):
            improved_in_r = False
            for subset in combinations(preceding_vars, r):
                parent_names = list(subset)
                cpt = variable.estimate_cpt_with_parents(parent_names, self.variables)
                score = variable.temp_BIC_score([self.find_variable(parent_name) for parent_name in parent_names], cpt)
                if score > best_score:
                    # print("Update Best Parents: ", [candidate_parent for candidate_parent in parent_names])
                    improved_in_r = True
                    best_score = score
                    best_parents = parent_names
                    best_cpt = cpt
            if not improved_in_r:
                break
            
        """
        # check likelihood ratio
        LL = self.calculate_log_likelihood(variable)
        likelihood_ratio = (LL0 - LL) / LL0
        print("Likelihood Ratio: ", likelihood_ratio)
        """
        return best_parents, best_cpt
    
    def set_optimal_parents(self, variable, preceding_vars, used_row=None):
        # print("Finding optimal parents for variable: ", variable.name)
        best_parents = []
        best_score = float('-inf')
        
        LL0 = self.calculate_LL0(variable)
        
        for r in range(min(len(preceding_vars) + 1, 3)):
            improved_in_r = False
            for subset in combinations(preceding_vars, r):
                candidate_parents = [self.variables[var] for var in subset]
                variable.set_parents(candidate_parents)
                variable.estimate_cpt()

                score = variable.BIC_sep()

                # print("Candidate Parents: ", [self.variables[var].name for var in subset])
                # print("Score: ", score)

                if score > best_score:
                    improved_in_r = True
                    # print("Update Best Parents: ", [candidate_parent.name for candidate_parent in candidate_parents])
                    best_score = score
                    best_parents = candidate_parents
            if not improved_in_r:
                break
        
        variable.set_parents(best_parents)
        variable.estimate_cpt()

        # check likelihood ratio
        LL = variable.calculate_log_likelihood()
        likelihood_ratio = (LL0 - LL) / LL0
        # print("Likelihood Ratio: ", likelihood_ratio)

    # another structure learning method, most simple greedy
    def greedy_structure_learning(self, max_iterations=10000):
        start_time = time.time()
        self.randomize_parent_sets()
        improvement = True
        iteration = 0
        while improvement and iteration < max_iterations:
            improvement = False
            best_gain = 0
            best_operation = None

            # Evaluate all possible legal arc operations
            for operation in self.get_legal_arc_operations():
                gain = self.calculate_bic_gain(operation)
                if gain > best_gain:
                    best_gain = gain
                    best_operation = operation

            # Perform the best operation, if any
            if best_gain > 0:
                # print("Best Operation: ", best_operation)
                self.perform_arc_operation(best_operation)
                improvement = True
            iteration += 1
        print(iteration)
        final_score = self.BIC_all()
        print("Final Score : ", final_score)
        self.calc_time = time.time() - start_time
        self.score = final_score

    def tabu_structure_learning(self, tabu_length=10, max_iterations=10000):
        start_time = time.time()
        self.randomize_parent_sets()
        improvement = True
        iteration = 0
        tabu_list = []

        while improvement and iteration < max_iterations:
            improvement = False
            best_gain = 0
            best_operation = None

            # Evaluate all possible legal arc operations not in the Tabu list
            for operation in self.get_legal_arc_operations():
                if operation not in tabu_list:
                    gain = self.calculate_bic_gain(operation)
                    if gain > best_gain:
                        best_gain = gain
                        best_operation = operation

            # Perform the best operation, if any
            if best_gain > 0:
                self.perform_arc_operation(best_operation)
                improvement = True

                # Add the reverse of the operation to the Tabu list
                reverse_operation = self.get_reverse_operation(best_operation)
                tabu_list.append(reverse_operation)

                # Keep the Tabu list within the specified length
                if len(tabu_list) > tabu_length:
                    tabu_list.pop(0)

            iteration += 1
        print(iteration)
        final_score = self.BIC_all()
        print("Final Score : ", final_score)
        self.calc_time = time.time() - start_time
        self.score = final_score

    # functions used in structure learning
    def get_reverse_operation(self, operation):
        # Return the reverse of the given operation
        var_name, parent_name, op_type = operation
        if op_type == "add":
            return (var_name, parent_name, "remove")
        elif op_type == "remove":
            return (var_name, parent_name, "add")
        elif op_type == "reverse":
            return (parent_name, var_name, "reverse")  # Reverse the direction
        else:
            raise ValueError("Invalid operation type.")

    def get_legal_arc_operations(self):
        # get all possible arc operations (add, remove, reverse)
        # operations are like (A, B, "add"), (A, B, "remove"), (A, B, "reverse")
        operations = []
        for var_name in self.variables.keys():
            for parent_name in self.variables.keys():
                if var_name != parent_name:
                    if self.check_legal_arc_operation(var_name, parent_name, "add"):
                        operations.append((var_name, parent_name, "add"))
                    if self.check_legal_arc_operation(var_name, parent_name, "remove"):
                        operations.append((var_name, parent_name, "remove"))
                    if self.check_legal_arc_operation(var_name, parent_name, "reverse"):
                        operations.append((var_name, parent_name, "reverse"))
            
        return operations
    
    def check_legal_arc_operation(self, var_name, parent_name, operation):
        # check if the arc operation is legal
        # operation is like "add", "remove", "reverse"
        if operation == "add":
            if var_name in self.variables[parent_name].parents:
                return False
            else:
                return not self.reachable(var_name, parent_name)
        elif operation == "remove":
            if var_name in self.variables[parent_name].parents:
                return True
            else:
                return False
        elif operation == "reverse":
            if var_name in self.variables[parent_name].parents:
                return not self.reachable(parent_name, var_name, allowDirect=False)
            else:
                return False
        else:
            raise ValueError("Invalid operation. Choose 'add', 'remove', or 'reverse'.")
    
    # check if start_node is reachable from end_node by going up to the parents
    def reachable(self, start_node, end_node, allowDirect=True, visited=None):
        if len(self.find_variable(end_node).parents) == 0:
            return False
        
        if visited is None:
            visited = set()

        visited.add(end_node)
        
        for parent in self.find_variable(end_node).parents:
            if parent.name == start_node:
                if not allowDirect:
                    continue
                else:
                    return True
            elif parent.name in visited:
                if self.reachable(start_node, parent.name, allowDirect=True):
                    return True

        return False
    
    # calculate the BIC gain
    def calculate_bic_gain(self, operation):
        if operation[2] == "add":
            score = -1 * self.variables[operation[0]].BIC_sep()
            self.variables[operation[0]].parents.append(self.variables[operation[1]])
            self.variables[operation[0]].estimate_cpt()
            score += self.variables[operation[0]].BIC_sep()
            self.variables[operation[0]].parents.remove(self.variables[operation[1]])
            self.variables[operation[0]].estimate_cpt()
            return score
        elif operation[2] == "remove":
            score = -1 * self.variables[operation[0]].BIC_sep()
            self.variables[operation[0]].parents.remove(self.variables[operation[1]])
            self.variables[operation[0]].estimate_cpt()
            score += self.variables[operation[0]].BIC_sep()
            self.variables[operation[0]].parents.append(self.variables[operation[1]])
            self.variables[operation[0]].estimate_cpt()
            return score
        elif operation[2] == "reverse":
            score = -1 * (self.variables[operation[0]].BIC_sep() + self.variables[operation[1]].BIC_sep())
            self.variables[operation[0]].parents.remove(self.variables[operation[1]])
            self.variables[operation[1]].parents.append(self.variables[operation[0]])
            self.variables[operation[0]].estimate_cpt()
            self.variables[operation[1]].estimate_cpt()
            score += self.variables[operation[0]].BIC_sep() + self.variables[operation[1]].BIC_sep()
            self.variables[operation[0]].parents.append(self.variables[operation[1]])
            self.variables[operation[1]].parents.remove(self.variables[operation[0]])
            self.variables[operation[0]].estimate_cpt()
            self.variables[operation[1]].estimate_cpt()
            return score
        else:
            raise ValueError("Invalid operation. Choose 'add', 'remove', or 'reverse'.")

    # perform the arc operation
    def perform_arc_operation(self, operation):
        if operation[2] == "add":
            self.variables[operation[0]].parents.append(self.variables[operation[1]])
            self.variables[operation[0]].estimate_cpt()
        elif operation[2] == "remove":
            self.variables[operation[0]].parents.remove(self.variables[operation[1]])
            self.variables[operation[0]].estimate_cpt()
        elif operation[2] == "reverse":
            self.variables[operation[0]].parents.remove(self.variables[operation[1]])
            self.variables[operation[1]].parents.append(self.variables[operation[0]])
            self.variables[operation[0]].estimate_cpt()
            self.variables[operation[1]].estimate_cpt()
        else:
            raise ValueError("Invalid operation. Choose 'add', 'remove', or 'reverse'.")
        
    def randomize_parent_sets(self):
        variable_names = list(self.variables.keys())
        random.shuffle(variable_names)
        searched_variables = []
        for var_name in variable_names:
            self.variables[var_name].parents = []
        for var_name in variable_names:
            variable = self.variables[var_name]

            # Randomly decide the number of parents (you can set limits as needed)
            num_parents = random.randint(0, min(len(searched_variables), 5))

            # Randomly select parent variables
            potential_parents = [name for name in searched_variables]
            selected_parents = random.sample(potential_parents, num_parents)

            # Assign these parents to the variable
            if len(selected_parents) > 0:
                variable.set_parents([self.variables[parent_name] for parent_name in selected_parents])
            variable.estimate_cpt()
            searched_variables.append(var_name)

    # generate data
    def generate(self, num_samples, start_node=None):
        update_ordering = self.ordering.copy()
        if start_node is not None:
            update_ordering = update_ordering[update_ordering.index(start_node)+1:]
        for var_name in update_ordering:
            self.variables[var_name].generate(num_samples)

        return None
    
    def set_random_cpt(self):
        for var in self.variables.values():
            var.set_random_cpt()

    # Setting data to each variable
    def set_data_from_dataloader(self, dataloader, column_list=None, dataRange=None):
        if column_list == None:
            column_list = list(self.variables.keys())
        variables = dataloader.get_data(column_list, dataRange=dataRange)
        for name, variable in variables.items():
            inVar = self.find_variable(name)
            if inVar is not None:
                inVar.set_data(variable.get_data('input'), name)
            else:
                # print(f"Warning: Variable {name} not found in ObjectNode {self.name}. Creating new variable.")
                self.add_variable(variable)
                self.ordering.append(name)

    # Evaluate performance
    def evaluate(self, targetVar, controlVar=None, changeRate=0.01, type="log_likelihood", num_samples=1000, tryTime=1, rand=None, folderPath=None):

        target_variable = self.find_variable(targetVar)
        control_variable = self.find_variable(controlVar) if controlVar else None

        if type == "log_likelihood":
            # sum up log likelihood for every variable
            return self.ll_all()
        if type == "BIC":
            # sum up BIC score for every variable
            return self.BIC_all()
        elif type == "elasticity":
            if rand is None:
                rand = np.random.rand(len(target_variable.get_data('input')))
                np.savetxt(os.path.join(folderPath, "rand.csv"), rand, delimiter=",")
            if self.reachable(control_variable.name, target_variable.name):
                return np.mean([self.calculate_elasticity(target_variable, control_variable, changeRate, num_samples, rand) for _ in range(tryTime)])
            else:
                return 0

        # check elasticity
        """try:
            elasticity = target_variable.elasticity(change_rate)
        except:
            print("Error: Could not calculate elasticity.")
            print("This is mainly because the variable has no parents.")
        print("Elasticity: ", elasticity)"""

    def calculate_mutual_information(self, q_var_name, p_var_name):
        # check if q_var is reachable from p_var
        if not self.reachable(p_var_name, q_var_name):
            return 0
        # Calculate the mutual information between two variables
        q_var = self.find_variable(q_var_name)
        p_var = self.find_variable(p_var_name)
        q_data = q_var.get_data('input')
        p_data = p_var.get_data('input')
        q_states = q_var.get_states('input')
        p_states = p_var.get_states('input')

        # Calculate the joint distribution
        joint_dist = np.zeros((q_states, p_states))
        for i in range(len(q_data)):
            joint_dist[q_data[i], p_data[i]] += 1

        # Calculate the marginal distributions
        q_marginal = np.sum(joint_dist, axis=1)
        p_marginal = np.sum(joint_dist, axis=0)

        joint_dist = joint_dist / len(q_data)
        q_marginal = q_marginal / len(q_data)
        p_marginal = p_marginal / len(q_data)

        # Calculate the mutual information
        mutual_information = 0
        for i in range(q_states):
            for j in range(p_states):
                if joint_dist[i, j] > 0:
                    mutual_information += joint_dist[i, j] * math.log2(joint_dist[i, j] / (q_marginal[i] * p_marginal[j]))

        return mutual_information

    # Display the optimized structure
    def visualize_structure(self):
        G = nx.DiGraph()
        pos = {}  # Dictionary to hold position data
        row_col = [0, 0]  # To keep track of row and column, used as mutable type for recursive modification

        def draw_node(variables, ordering):
            nonlocal row_col
            for var_name in ordering:
                variable = variables[var_name]
                if isinstance(variable, ObjectNode):
                    draw_node(variable.variables, variable.ordering)
                else:
                    G.add_node(var_name)
                    pos[var_name] = (row_col[1], -row_col[0])  # Set the position for this node

                    for parent in variable.parents:
                        G.add_edge(parent.output, var_name)

                    row_col[1] += 1
                    if row_col[1] >= 3:  # Maximum number of columns
                        row_col[1] = 0
                        row_col[0] += 1

        draw_node(self.variables, self.ordering)

        nx.draw(G, pos, with_labels=True, node_color="lightblue", font_size=16, node_size=700, font_color="black", font_weight="bold", arrowsize=20)
        plt.title("Bayesian Network Structure")
        plt.axis("off")
        plt.show()

    # saving functions
    def save_data(self, filename, file_type="csv"):
        datatable = np.concatenate([self.variables[varname].tabledata() for varname in self.ordering], axis=1)
        dataframe = pd.DataFrame(datatable, columns=self.ordering)
        if file_type == "csv":
            dataframe.to_csv(os.path.join("data", "midData", filename+"."+file_type), index=False)

    def tabledata(self):
        return np.concatenate([self.variables[varname].tabledata() for varname in self.ordering], axis=1)
    
    def save_model_parameters(self, filename):
        model_params = self._extract_model_params()
        with open(os.path.join("data", "modelData", filename+".json"), 'w') as file:
            json.dump(model_params, file, indent=4)

    def _extract_model_params(self, model_params=None):
        if model_params is None:
            model_params = {"variables": {}, "objects": {}, "score": self.score, "timeTaken": 0}
        model_params["timeTaken"] += self.calc_time
        model_params["objects"][self.name] = {}
        model_params["objects"][self.name]["variables"] = []
        model_params["objects"][self.name]["in_obj"] = []
        for var_name in self.ordering:
            variable = self.variables[var_name]
            if isinstance(variable, ObjectNode):
                # If the variable is an ObjectNode, recursively extract its parameters
                model_params = variable._extract_model_params(model_params=model_params)
                model_params["objects"][self.name]["in_obj"].append(var_name)
            else:
                # Otherwise, extract the variable's parameters as usual
                model_params["variables"][var_name] = {
                    "num_states": variable.states,
                    "parents": [parent.output for parent in variable.parents],
                    "cpt": variable.cpt.tolist() if variable.cpt is not None else None,
                    "BIC": variable.BIC_sep(),
                }
                model_params["objects"][self.name]["variables"].append(var_name)
        return model_params
    

    def calculate_elasticity(self, target_variable, control_variable, change_rate, num_samples=10000, rand=None):
        # generate data with original condition
        # self.generate(num_samples=num_samples, start_node=control_variable.name)
        prob_table_ori = self.aggregate_distribution_table(target_variable, num_samples)

        # modify control variable's data
        control_variable.modify_data(change_rate, rand=rand)

        # generate data with modified condition
        # self.generate(num_samples=num_samples, start_node=control_variable.name)
        prob_table_mod = self.aggregate_distribution_table(target_variable, num_samples)

        # calculate elasticity
        elasticity = np.mean(np.abs(prob_table_ori - prob_table_mod) / prob_table_ori)

        return elasticity
    
    def aggregate_distribution_table(self, target_variable, num_samples=10000):
        # get the distribution table of the target variable
        target_probs = target_variable.probability_array(num_samples=num_samples)
        target_dist = np.mean(target_probs, axis=0)

        return target_dist


if __name__=="__main__":
    # Test code
    np.random.seed(0)
    data_A = np.random.choice([0, 1], size=100)
    data_B = np.random.choice([0, 1], size=100)
    data_C = np.random.choice([0, 1], size=100)
    data_D = np.random.choice([0, 1], size=100)

    A = Variable("A", 2)
    B = Variable("B", 2)
    C = Variable("C", 2)
    D = Variable("D", 2)

    A.set_data(data_A)
    B.set_data(data_B)
    C.set_data(data_C)
    D.set_data(data_D)

    engine = ObjectNode("Engine")

    engine.add_variable(A)
    engine.add_variable(B)
    engine.add_variable(C)
    engine.add_variable(D)

    print("Estimation starts")

    fixed_positions = {'D': 0, 'C': 3}
    engine.order_optimization(fixed_positions=fixed_positions)

    # Display the optimized structure
    for var_name, variable in engine.variables.items():
        parent_names = [parent.name for parent in variable.parents]
        print(f"Variable {var_name} has parents {parent_names}")