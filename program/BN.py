import numpy as np

class Variable:
    def __init__(self, name, states):
        self.name = name
        self.states = states  # The number of possible states this variable can take
        self.parents = []  # List to store parent variables
        self.cpt = None  # Conditional Probability Table as a NumPy array

    def set_parents(self, parents):
        self.parents = parents

    def set_cpt(self, cpt_array):
        self.cpt = np.array(cpt_array)

    def probability(self, parent_states):
        # Create an index tuple to access the correct element in the CPT NumPy array
        index = tuple(parent_states.tolist())
        
        # Fetch the probability from the CPT using the index
        return self.cpt[index]


if __name__=="__main__":
    # Create variables A, B, and C with 2 states each
    A = Variable("A", 2)
    B = Variable("B", 2)
    C = Variable("C", 2)

    # Set parents for C
    C.set_parents([A, B])

    # Create a CPT for C using a NumPy array
    # This is a 2x2x2 array representing P(C | A, B)
    # The last dimension represents the state of C, while the first and second dimensions represent the states of A and B
    cpt_C = np.array([[[0.1, 0.9], [0.4, 0.6]], [[0.7, 0.3], [0.2, 0.8]]])

    # Set the CPT for C
    C.set_cpt(cpt_C)

    # Calculate probabilities
    parent_states_A0_B0 = np.array([0, 0])
    prob_C_given_A0_B0 = C.probability(parent_states_A0_B0)

    parent_states_A1_B1 = np.array([1, 1])
    prob_C_given_A1_B1 = C.probability(parent_states_A1_B1)

    print(f"P(C| A=0, B=0) = {prob_C_given_A0_B0}")
    print(f"P(C| A=1, B=1) = {prob_C_given_A1_B1}")

