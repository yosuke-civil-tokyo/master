import numpy as np
import math

class Variable:
    def __init__(self, name, states):
        self.name = name
        self.states = states  # The number of possible states this variable can take
        self.parents = []  # List to store parent variables
        self.data = None  # Field to store data as a NumPy array
        self.cpt = None  # Conditional Probability Table as a NumPy array
        self.object_node = None  # Field to store an ObjectNode, if this variable belongs to on

    def set_parents(self, parents):
        self.parents = parents

    def set_cpt(self, cpt_array):
        self.cpt = np.array(cpt_array)

    def set_data(self, data_array, name=None):
        self.data = np.array(data_array)
        if name:
            self.name = name

    def get_data(self, input_or_output='input'):
        return self.data
    
    def get_states(self, input_or_output='input'):
        return self.states

    # this is for future use, after vectorization
    def probability(self, parent_data=None):
        # use table concatenation of self.data in parent variables, is parent_data is None
        if parent_data is None:
            parent_data = np.stack([parent.get_data('output') for parent in self.parents], axis=-1)
        # Create an index tuple to access the correct slice in the CPT NumPy array
        prob = self.cpt[tuple(parent_data.tolist())]
        if np.sum(prob) == 0:
            return [1 / self.states] * self.states
        
        return prob
    
    def estimate_cpt(self):
        if self.get_data('input') is None:
            raise ValueError("Data not set for this variable.")
        
        num_states = [parent.get_states('output') for parent in self.parents] + [self.get_states('input')]
        
        # Initialize CPT with zeros
        self.cpt = np.zeros(num_states)
        
        # Compute the indices for each data point
        indices = np.vstack([parent.get_data('output') for parent in self.parents] + [self.get_data('input')])

        # Vectorized calculation of counts
        np.add.at(self.cpt, tuple(indices), 1)

        # Normalize to get probabilities
        self.cpt /= self.cpt.sum(axis=-1, keepdims=True)

    def estimate_cpt_with_parents(self, parent_names, variables_dict):
        parents = [variables_dict[name] for name in parent_names]

        num_states = [parent.get_states('output') for parent in parents] + [self.get_states('input')]
        
        # Initialize CPT with zeros
        cpt = np.zeros(num_states)
        
        # Compute the indices for each data point
        indices = np.vstack([parent.get_data('output') for parent in parents] + [self.get_data('input')])

        # Vectorized calculation of counts
        np.add.at(cpt, tuple(indices), 1)

        # Normalize to get probabilities
        cpt /= cpt.sum(axis=-1, keepdims=True)
        return cpt
        

    # to sample
    def generate(self, parent_states):
        # probability array
        prob = self.probability(parent_states)
        # sample from probability array
        return np.random.choice(self.states, p=prob)

    # evaluation functions
    # log likelihood
    def log_likelihood(self):
        data = self.get_data('input')
        if self.parents:
            # When there are parent variables
            indices = np.stack([parent.get_data('output') for parent in self.parents] + [data], 0)
            probs = self.cpt[tuple(indices)]
        else:
            # When there are no parent variables (independent variable)
            probs = self.cpt[data]

        log_likelihood = np.sum(np.nan_to_num(np.log(probs + 1e-6)))  # Adding a small constant to avoid log(0)
        return log_likelihood
    
    # elasticity of prediction
    def elasticity(self, change_rate=0.01):
        # array of parents' states
        original_data = np.concatenate([parent.get_data('output') for parent in self.parents], axis=-1).reshape((-1, len(self.parents)))
        random_data = np.concatenate(
            [np.random.choice(parent.states, size=len(parent.get_data('output'))) for parent in self.parents],
             axis=-1).reshape((-1, len(self.parents)))
        # modified_data = 

        modified_data = np.where((np.random.rand(len(original_data)) < change_rate).reshape((len(original_data), 1)), random_data, original_data)

        # prediction results from original data and modified data
        original_prediction = np.array([self.generate(d) for d in original_data])
        modified_prediction = np.array([self.generate(d) for d in modified_data])

        return np.mean(original_prediction != modified_prediction)
    





if __name__ == "__main__":
    # Simulated data for demonstration
    np.random.seed(0)
    data_A = np.random.choice([0, 1], size=1000)
    data_B = np.random.choice([0, 1], size=1000)
    data_C = np.random.choice([0, 1], size=1000)

    # Create Variables
    A = Variable("A", 2)
    B = Variable("B", 2)
    C = Variable("C", 2)

    # Set data
    A.set_data(data_A)
    B.set_data(data_B)
    C.set_data(data_C)

    # Set parents for C
    C.set_parents([A, B])

    # Estimate CPT for C
    C.estimate_cpt()

    print("Estimated CPT for C:", C.cpt)