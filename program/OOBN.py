import math
from itertools import chain, combinations
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

from BN import Variable


# ObjectNode class definition
class ObjectNode(Variable):
    def __init__(self, name, variables={}):
        super().__init__(name, states=None) 
        self.name = name
        self.variables = variables
        self.input = []
        self.input_data = self.data
        self.output_data = self.data
        self.input_states = self.states
        self.output_states = self.states
    

    def set_data(self, data_array, variable_name=None, data_type='input'):
        if data_type == 'input':
            self.input_data = np.array(data_array)
            self.input_states=int(np.max(data_array) + 1)
            self.input.append(variable_name)
        elif data_type == 'output':
            self.output_data = np.array(data_array)
            self.output_states=int(np.max(data_array) + 1)

    def get_variables(self, data_type='input'):
        if data_type == 'input':
            return {var_name: self.variables[var_name] for var_name in self.input}

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

    def structure_optimization(self, fixed_positions=None):
        # check if any variable has to be fixed
        if fixed_positions is None:
            fixed_positions = {}
    
        current_ordering = list(self.variables.keys())
        random.shuffle(current_ordering)

        # Apply fixed positions if provided
        for var, pos in fixed_positions.items():
            current_ordering.remove(var)
            current_ordering.insert(pos, var)

        best_score = float('-inf')

        while True:
            best_swap_index = None

            for i in range(len(current_ordering) - 1):

                # Skip over fixed positions
                if current_ordering[i] in fixed_positions.keys() or current_ordering[i+1] in fixed_positions.keys():
                    continue
                # Temporarily swap and update structure
                current_ordering[i], current_ordering[i + 1] = current_ordering[i + 1], current_ordering[i]
                self.update_structure(current_ordering)

                # Estimate CPT and calculate BIC score
                for var_name in current_ordering:
                    variable = self.variables[var_name]
                    if variable.parents:
                        variable.estimate_cpt()

                score = self.BIC_all()
                print(score)

                if score > best_score:
                    best_swap_index = i

                # Undo the swap
                current_ordering[i], current_ordering[i + 1] = current_ordering[i + 1], current_ordering[i]

            # If we found a swap that improves the score, perform the swap
            if best_swap_index is not None:
                current_ordering[best_swap_index], current_ordering[best_swap_index + 1] = current_ordering[best_swap_index + 1], current_ordering[best_swap_index]
                best_score = score
            else:
                self.ordering = current_ordering
                print("best ordering")
                self.update_structure(self.ordering)
                score = self.BIC_all()
                print("Final Score : ", score)
                break  # No improving swap was found

    def BIC_all(self):
        score = 0
        N = len(next(iter(self.variables.values())).get_data('input'))
        
        for variable in self.variables.values():
            k = variable.cpt.ndim
            log_likelihood = self.calculate_log_likelihood(variable)
            score += log_likelihood - (k / 2) * math.log(N)
        
        return score
    
    def BIC_sep(self, variable):
        N = len(variable.get_data('input'))
        
        k = variable.cpt.ndim
        log_likelihood = self.calculate_log_likelihood(variable)
        score = log_likelihood - (k / 2) * math.log(N)
        
        print("CPT size: ", k)
        print("Log Likelihood: ", log_likelihood)
        return score

    def calculate_log_likelihood(self, variable):
        log_likelihood = 0
        for i, val in enumerate(variable.get_data('input')):
            parent_states = np.array([parent.get_data('output')[i] for parent in variable.parents])
            prob = variable.probability(parent_states)[val]
            log_likelihood += math.log(prob)
        
        return log_likelihood
    
    def calculate_LL0(self, variable):
        # For a variable with k states, the log-likelihood under the null model is N * log(1/k)
        k = variable.get_states('input')
        N = len(variable.get_data('input'))
        log_likelihood = N * math.log(1 / k)
        return log_likelihood

    def update_structure(self, ordering):
        for var_name in ordering:
            variable = self.variables[var_name]
            # skip if the variable is input type

            self.find_optimal_parents(variable, ordering)

    def find_optimal_parents(self, variable, ordering):
        best_parents = []
        best_score = float('-inf')
        
        preceding_vars = ordering[:ordering.index(variable.name)]
        LL0 = self.calculate_LL0(variable)
        
        for subset in chain.from_iterable(combinations(preceding_vars, r) for r in range(len(preceding_vars) + 1)):
            candidate_parents = [self.variables[var] for var in subset]
            variable.set_parents(candidate_parents)
            variable.estimate_cpt()
            
            score = self.BIC_sep(variable)

            print("Candidate Parents: ", [self.variables[var].name for var in subset])
            print("Score: ", score)
            
            if score > best_score:
                print("Update Best Parents: ", [candidate_parent.name for candidate_parent in candidate_parents])
                best_score = score
                best_parents = candidate_parents
        
        variable.set_parents(best_parents)
        variable.estimate_cpt()

        # check likelihood ratio
        LL = self.calculate_log_likelihood(variable)
        likelihood_ratio = (LL0 - LL) / LL0
        print("Likelihood Ratio: ", likelihood_ratio)

    # Setting data to each variable
    def set_data_from_dataloader(self, dataloader, column_list):
        variables = dataloader.get_data(column_list)
        
        for name, variable in variables.items():
            if name in self.variables:
                self.variables[name].set_data(variable.get_data('input'), name)
            else:
                # print(f"Warning: Variable {name} not found in ObjectNode {self.name}. Creating new variable.")
                self.add_variable(variable)

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
                        G.add_edge(parent.name, var_name)

                    row_col[1] += 1
                    if row_col[1] >= 3:  # Maximum number of columns
                        row_col[1] = 0
                        row_col[0] += 1

        draw_node(self.variables, self.ordering)

        nx.draw(G, pos, with_labels=True, node_color="lightblue", font_size=16, node_size=700, font_color="black", font_weight="bold", arrowsize=20)
        plt.title("Bayesian Network Structure")
        plt.axis("off")
        plt.show()


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
    engine.structure_optimization(fixed_positions=fixed_positions)

    # Display the optimized structure
    for var_name, variable in engine.variables.items():
        parent_names = [parent.name for parent in variable.parents]
        print(f"Variable {var_name} has parents {parent_names}")