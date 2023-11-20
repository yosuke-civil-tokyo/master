import math
from itertools import chain, combinations
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from model.BN import Variable


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
        self.ordering = []
    
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
        improvement = True
        swap_pairs = [[i, i+1] for i in range(len(current_ordering) - 1) if not {i, i+1} & set(fixed_positions.values())]
        
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

    def evaluate_swap(self, pair, ordering):
        ordering = ordering.copy()
        ordering[pair[0]], ordering[pair[1]] = ordering[pair[1]], ordering[pair[0]]
        # update structure
        score = self.score_ordering(ordering)
        # score = self.BIC_all()
        return score, ordering

    def score_ordering(self, ordering):
        name_to_cpt = {}
        name_to_parents = {var_name: [] for var_name in self.variables.keys()}

        for var_name in ordering:
            preceding_vars = ordering[:ordering.index(var_name)]
            best_parents, best_cpt = self.find_optimal_parents(var_name, preceding_vars)
            name_to_parents[var_name] = best_parents
            name_to_cpt[var_name] = best_cpt

        total_score = sum(self.temp_BIC_score(var_name, name_to_parents, name_to_cpt[var_name]) for var_name in ordering)
        return total_score

    def temp_BIC_score(self, var_name, name_to_parents, cpt):
        # Calculate the BIC score for a given variable, its parents, and its CPT
        variable = self.variables[var_name]
        N = len(variable.get_data('input'))
        k = cpt.size
        log_likelihood = self.temp_calculate_log_likelihood(var_name, name_to_parents, cpt)
        score = log_likelihood - (k / 2) * math.log(N)
        return score

    def temp_calculate_log_likelihood(self, var_name, name_to_parents, cpt):
        # Implement the logic similar to calculate_log_likelihood but use the provided CPT and parent names
        variable = self.variables[var_name]
        data = variable.get_data('input')
        if name_to_parents[var_name]:
            indices = np.stack([self.variables[parent_name].get_data('output') for parent_name in name_to_parents[var_name]] + [data], 0)
            probs = cpt[tuple(indices)]
        else:
            probs = cpt[data]
        log_likelihood = np.sum(np.log(probs))
        return log_likelihood

    def BIC_all(self):
        score = 0
        
        for variable in self.variables.values():
            score += self.BIC_sep(variable)
        
        return score
    
    def BIC_sep(self, variable):
        # when the variable is an object node
        if isinstance(variable, ObjectNode):
            # calculate the score for each input variable in the object node
            score = 0
            for input_var_name in variable.input:
                input_variable = variable.variables[input_var_name]
                score += self.BIC_sep(input_variable)

        else:
            # when the variable is not an object node
            N = len(variable.get_data('input'))

            k = variable.cpt.size
            log_likelihood = self.calculate_log_likelihood(variable)
            score = log_likelihood - (k / 2) * math.log(N)
            
            # print("CPT size: ", k)
            # print("Log Likelihood: ", log_likelihood)

        return score

    def calculate_log_likelihood(self, variable):
        data = variable.get_data('input')
        if variable.parents:
            # When there are parent variables
            indices = np.stack([parent.get_data('output') for parent in variable.parents] + [data], 0)
            # get the probability of each data point
            probs = variable.cpt[tuple(indices)]
        else:
            # When there are no parent variables (independent variable)
            probs = variable.cpt[data]

        log_likelihood = np.sum(np.log(probs))
        return log_likelihood
    
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
        print("Finding optimal parents for variable: ", variable)
        best_parents = []
        best_cpt = None
        best_score = float('-inf')
        
        LL0 = self.calculate_LL0(self.variables[variable])
        
        for subset in chain.from_iterable(combinations(preceding_vars, r) for r in range(len(preceding_vars) + 1)):
            parent_names = list(subset)
            cpt = self.variables[variable].estimate_cpt_with_parents(parent_names, self.variables)
            score = self.temp_BIC_score(variable, {variable: parent_names}, cpt)
            if score > best_score:
                print("Update Best Parents: ", [candidate_parent for candidate_parent in parent_names])
                best_score = score
                best_parents = parent_names
                best_cpt = cpt
            
        """
        # check likelihood ratio
        LL = self.calculate_log_likelihood(variable)
        likelihood_ratio = (LL0 - LL) / LL0
        print("Likelihood Ratio: ", likelihood_ratio)
        """
        return best_parents, best_cpt
    
    def set_optimal_parents(self, variable, preceding_vars):
        print("Finding optimal parents for variable: ", variable.name)
        best_parents = []
        best_score = float('-inf')
        
        LL0 = self.calculate_LL0(variable)
        
        for subset in chain.from_iterable(combinations(preceding_vars, r) for r in range(len(preceding_vars) + 1)):
            candidate_parents = [self.variables[var] for var in subset]
            variable.set_parents(candidate_parents)
            variable.estimate_cpt()
            
            score = self.BIC_sep(variable)

            # print("Candidate Parents: ", [self.variables[var].name for var in subset])
            # print("Score: ", score)
            
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

    # generate data
    def generate(self, num_samples):
        for var_name in self.ordering:
            self.variables[var_name].generate(num_samples)

        return None

    # Setting data to each variable
    def set_data_from_dataloader(self, dataloader, column_list):
        variables = dataloader.get_data(column_list)
        for name, variable in variables.items():
            if name in self.variables:
                self.variables[name].set_data(variable.get_data('input'), name)
            else:
                # print(f"Warning: Variable {name} not found in ObjectNode {self.name}. Creating new variable.")
                self.add_variable(variable)

    # Evaluate performance
    def evaluate(self, target_variable_name, change_rate=0.01):
        print("Evaluating performance...")
        print("Target Variable: ", target_variable_name)
        target_variable = self.variables[target_variable_name]

        # check log-likelihood
        ll = target_variable.log_likelihood()
        print("Log Likelihood: ", ll)

        # check elasticity
        try:
            elasticity = target_variable.elasticity(change_rate)
        except:
            print("Error: Could not calculate elasticity.")
            print("This is mainly because the variable has no parents.")
        print("Elasticity: ", elasticity)

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