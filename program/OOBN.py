import math
from itertools import chain, combinations
import numpy as np
import random

from BN import Variable


# ObjectNode class definition
class ObjectNode:
    def __init__(self, name):
        self.name = name
        self.variables = {}
    
    def add_variable(self, variable):
        self.variables[variable.name] = variable
        variable.object_node = self

    def structure_optimization(self):
        current_ordering = list(self.variables.keys())
        random.shuffle(current_ordering)

        best_score = float('-inf')

        while True:
            max_improvement = float('-inf')
            best_swap_index = None

            for i in range(len(current_ordering) - 1):
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
            print(variable.cpt)
            
            score = self.BIC_sep(variable)
            
            if score > best_score:
                best_score = score
                best_parents = candidate_parents
        
        variable.set_parents(best_parents)
        variable.estimate_cpt()

    # setting data to each variable
    def set_data_from_dataloader(self, dataloader):
        data, variable_names = dataloader.get_data()
        
        for name, values in data.items():
            if name in self.variables:
                self.variables[name].set_data(values, name)
            else:
                print(f"Warning: Variable {name} not found in ObjectNode {self.name}. Creating new variable.")
                new_var = Variable(name, states=max(values) + 1)
                new_var.set_data(values, name)
                self.add_variable(new_var)

    


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

    engine.structure_optimization()

    # Display the optimized structure
    for var_name, variable in engine.variables.items():
        parent_names = [parent.name for parent in variable.parents]
        print(f"Variable {var_name} has parents {parent_names}")