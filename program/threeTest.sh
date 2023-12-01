#!/bin/bash

# for running three algorithms
# Run the structure_optimization function
python test/TimeTest.py model2-obj structure_optimization
python test/TimeTest.py model2-normal structure_optimization

# Run the greedy_structure_learning function
python test/TimeTest.py model2-normal greedy_structure_learning

# Run the tabu_structure_learning function
python test/TimeTest.py model2-normal tabu_structure_learning
