#!/bin/bash

# for running three algorithms
# Run the structure_optimization function
python test/TimeTest.py model1 structure_optimization

# Run the greedy_structure_learning function
python test/TimeTest.py model1 greedy_structure_learning

# Run the tabu_structure_learning function
python test/TimeTest.py model1 tabu_structure_learning
