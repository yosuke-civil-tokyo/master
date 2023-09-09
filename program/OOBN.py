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
        self.name = name
        self.variables = variables
    
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
        print(current_ordering)

        best_score = float('-inf')

        while True:
            max_improvement = float('-inf')
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

                # Calculate improvement
                improvement = score - best_score

                if improvement > max_improvement:
                    max_improvement = improvement
                    best_swap_index = i

                # Undo the swap
                current_ordering[i], current_ordering[i + 1] = current_ordering[i + 1], current_ordering[i]

            # If we found a swap that improves the score, perform the swap
            if best_swap_index is not None:
                current_ordering[best_swap_index], current_ordering[best_swap_index + 1] = current_ordering[best_swap_index + 1], current_ordering[best_swap_index]
                best_score += max_improvement
            else:
                break  # No improving swap was found

    def BIC_all(self):
        score = 0
        N = len(next(iter(self.variables.values())).data)
        
        for variable in self.variables.values():
            k = len(variable.cpt)
            log_likelihood = self.calculate_log_likelihood(variable)
            score += log_likelihood - (k / 2) * math.log(N)
        
        return score
    
    def BIC_sep(self, variable):
        N = len(variable.data)
        
        k = len(variable.cpt)
        log_likelihood = self.calculate_log_likelihood(variable)
        score = log_likelihood - (k / 2) * math.log(N)
        
        return score

    def calculate_log_likelihood(self, variable):
        log_likelihood = 0
        for i, val in enumerate(variable.data):
            parent_states = np.array([parent.data[i] for parent in variable.parents])
            prob = variable.probability(parent_states)[val]
            log_likelihood += math.log(prob)
        
        return log_likelihood

    def update_structure(self, ordering):
        for var_name in ordering:
            variable = self.variables[var_name]
            self.find_optimal_parents(variable, ordering)

    def find_optimal_parents(self, variable, ordering):
        best_parents = []
        best_score = float('-inf')
        
        preceding_vars = ordering[:ordering.index(variable.name)]
        
        for subset in chain.from_iterable(combinations(preceding_vars, r) for r in range(len(preceding_vars) + 1)):
            candidate_parents = [self.variables[var] for var in subset]
            variable.set_parents(candidate_parents)
            variable.estimate_cpt()
            
            score = self.BIC_sep(variable)
            
            if score > best_score:
                best_score = score
                best_parents = candidate_parents
        
        variable.set_parents(best_parents)
        variable.estimate_cpt()

    # Setting data to each variable
    def set_data_from_dataloader(self, dataloader, column_list):
        variables = dataloader.get_data(column_list)
        
        for name, variable in variables.items():
            if name in self.variables:
                self.variables[name].set_data(variable.data, name)
            else:
                # print(f"Warning: Variable {name} not found in ObjectNode {self.name}. Creating new variable.")
                self.add_variable(variable)

    # Display the optimized structure
    def visualize_structure(self):
        G = nx.DiGraph()
        
        for var_name, variable in self.variables.items():
            G.add_node(var_name)
            for parent in variable.parents:
                G.add_edge(parent.name, var_name)
        
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, with_labels=True, node_color="lightblue", font_size=16, node_size=700, font_color="black", font_weight="bold", arrowsize=20)
        plt.title("Bayesian Network Structure")
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