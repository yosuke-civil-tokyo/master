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
        index = tuple(parent_data.tolist())
        # Fetch the probabilities from the CPT using the index
        return self.cpt[index]
    
    def estimate_cpt(self):
        if self.get_data('input') is None:
            raise ValueError("Data not set for this variable.")
        
        num_states = [parent.get_states('output') for parent in self.parents] + [self.get_states('input')]
        
        # Initialize CPT with zeros
        self.cpt = np.zeros(num_states)
        
        # Compute the dimensions of the parent data
        parent_dims = [parent.get_data('output') for parent in self.parents]
        
        # Calculate joint counts
        for i, val in enumerate(self.get_data('input')):
            index = tuple(parent[i] for parent in parent_dims) + (val,)
            self.cpt[index] += 1
        
        # Normalize to get probabilities
        norm = self.cpt.sum(axis=-1, keepdims=True)
        np.place(norm, norm == 0, 1)
        self.cpt /= norm

    # to sample
    def generate(self, parent_states):
        # probability array
        prob = self.probability(parent_states)
        # sample from probability array
        return np.random.choice(self.states, p=prob)

    # to calculate score
    # log likelihood
    def log_likelihood(self, parent_states=None):
        log_likelihood = 0
        for i, val in enumerate(self.get_data('input')):
            parent_states = np.array([parent.get_data('output')[i] for parent in self.parents])
            prob = self.probability(parent_states)[val]
            log_likelihood += math.log(prob)
        
        return log_likelihood




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