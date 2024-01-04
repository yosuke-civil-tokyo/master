import numpy as np
import math

class Variable:
    def __init__(self, name, states):
        self.name = name
        self.output = name
        self.states = states  # The number of possible states this variable can take
        self.parents = []  # List to store parent variables
        self.data = None  # Field to store data as a NumPy array
        self.cpt = None  # Conditional Probability Table as a NumPy array
        self.object_node = None  # Field to store an ObjectNode, if this variable belongs to on

    def set_parents(self, parents):
        self.parents = parents

    def set_cpt(self, cpt_array):
        self.cpt = np.array(cpt_array)
        self.cpt = np.nan_to_num(self.cpt)

    def set_data(self, data_array, name=None):
        self.data = np.array(data_array)
        if name:
            self.name = name

    def get_data(self, input_or_output='input'):
        return self.data
    
    def get_states(self, input_or_output='input'):
        return self.states

    # this is for future use, after vectorization
    def probability_array(self, parent_data=None, num_samples=None):
        if parent_data:
            # When there are parent variables
            probs_array = self.cpt[tuple(parent_data)]
        elif self.parents:
            # When there are parent variables
            indices = np.stack([parent.get_data('output') for parent in self.parents], 0)
            probs_array = self.cpt[tuple(indices)]
        else:
            # When there are no parent variables (independent variable)
            probs_array = np.tile(self.cpt, (num_samples, 1))
        
        return probs_array
    
    def probability(self, parent_data=None, num_samples=None):
        data = self.get_data('input')
        probs_array = self.probability_array(parent_data=parent_data, num_samples=num_samples)
        probs = probs_array[tuple(data)]
        
        return probs
    
    def estimate_cpt(self):
        if self.get_data('input') is None:
            raise ValueError("Data not set for this variable.")
        
        num_states = [parent.get_states('output') for parent in self.parents] + [self.get_states('input')]
        
        # Initialize CPT with zeros
        self.cpt = np.zeros(num_states)
        
        # Compute the indices for each data point
        if hasattr(self, 'use_row'):
            indices = np.vstack([parent.get_data('output')[self.use_row] for parent in self.parents] + [self.get_data('input')[self.use_row]])
        else:
            indices = np.vstack([parent.get_data('output') for parent in self.parents] + [self.get_data('input')])

        # Vectorized calculation of counts
        np.add.at(self.cpt, tuple(indices), 1)

        # Normalize to get probabilities
        self.cpt /= self.cpt.sum(axis=-1, keepdims=True)
        self.cpt = np.nan_to_num(self.cpt)

    def estimate_cpt_with_parents(self, parents, variables_dict):
        num_states = [parent.get_states('output') for parent in parents] + [self.get_states('input')]
        
        # Initialize CPT with zeros
        cpt = np.zeros(num_states)
        
        # Compute the indices for each data point
        if hasattr(self, 'use_row'):
            indices = np.vstack([parent.get_data('output')[self.use_row] for parent in parents] + [self.get_data('input')[self.use_row]])
        else:
            indices = np.vstack([parent.get_data('output') for parent in parents] + [self.get_data('input')])

        # Vectorized calculation of counts
        np.add.at(cpt, tuple(indices), 1)

        # Normalize to get probabilities
        cpt /= cpt.sum(axis=-1, keepdims=True)
        cpt = np.nan_to_num(cpt)
        return cpt

    def BIC_sep(self):
        if hasattr(self, 'use_row'):
            N = np.sum(self.use_row)
        else:
            N = len(self.get_data('input'))
        k = self.cpt.size
        log_likelihood = self.calculate_log_likelihood()
        score = log_likelihood - (k / 2) * math.log(N)

        return score

    def ll_sep(self):
        score = self.calculate_log_likelihood()

        return score

    def calculate_log_likelihood(self):
        if hasattr(self, 'use_row'):
            data = self.get_data('input')[self.use_row]
        else:
            data = self.get_data('input')
        if self.parents:
            # When there are parent variables
            if hasattr(self, 'use_row'):
                indices = np.stack([parent.get_data('output')[self.use_row] for parent in self.parents] + [data], 0)
            else:
                indices = np.stack([parent.get_data('output') for parent in self.parents] + [data], 0)
            # get the probability of each data point
            probs = self.cpt[tuple(indices)]
        else:
            # When there are no parent variables (independent variable)
            probs = self.cpt[data]

        log_likelihood = np.sum(np.log(probs + 1e-6))
        return log_likelihood

    def temp_BIC_score(self, parents, cpt):
        if hasattr(self, 'use_row'):
            N = np.sum(self.use_row)
        else:
            N = len(self.get_data('input'))
        k = cpt.size
        log_likelihood = self.temp_log_likelihood(parents, cpt)
        score = log_likelihood - (k / 2) * math.log(N)

        return score

    def temp_log_likelihood(self, parents, cpt):
        if hasattr(self, 'use_row'):
            data = self.get_data('input')[self.use_row]
        else:
            data = self.get_data('input')
        if parents:
            # When there are parent variables
            if hasattr(self, 'use_row'):
                indices = np.stack([parent.get_data('output')[self.use_row] for parent in parents] + [data], 0)
            else:
                indices = np.stack([parent.get_data('output') for parent in parents] + [data], 0)
            # get the probability of each data point
            probs = cpt[tuple(indices)]
        else:
            # When there are no parent variables (independent variable)
            probs = cpt[data]

        log_likelihood = np.sum(np.log(probs + 1e-6))
        return log_likelihood
        
    # to sample
    def predict(self, parent_data=None):
        # probability array
        prob = self.probability(parent_data=parent_data)
        # sample from probability array
        return np.random.choice(self.states, p=prob)
    
    def generate(self, num_samples):
        probs = self.probability_array(num_samples=num_samples)
        # replace rows of nan, 0
        zero_rows = np.all(probs==0, axis=1)
        probs[zero_rows] = np.full((self.states,), 1/self.states)
        self.data = np.array([np.random.choice(self.states, p=probs[i]) for i in range(num_samples)])

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
        modified_data = np.where((np.random.rand(len(original_data)) < change_rate).reshape((len(original_data), 1)), random_data, original_data)

        # prediction results from original data and modified data
        original_prediction = np.array([self.predict(d) for d in original_data])
        modified_prediction = np.array([self.predict(d) for d in modified_data])

        return np.mean(original_prediction != modified_prediction)
    
    def tabledata(self):
        return self.data.reshape((-1, 1))
    
    def set_random_cpt(self):
        num_states = [parent.get_states('output') for parent in self.parents] + [self.get_states('input')]
        cpt = np.random.rand(*num_states)
        cpt /= cpt.sum(axis=-1, keepdims=True)
        self.cpt = np.nan_to_num(cpt)
        return cpt
    
    def modify_data(self, change_rate, rand=None):
        if rand is None:
            rand = np.random.rand(len(self.get_data('input')))
        original_data = self.get_data('input')
        change_data = np.where((original_data-1)==-1, self.states-1, original_data-1)
        modified_data = np.where(rand < change_rate, change_data, original_data)
        self.set_data(modified_data)
        
    





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