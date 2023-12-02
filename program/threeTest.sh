#!/bin/bash

# for running three algorithms
# Run the order_optimization function
python test/TimeTest.py model2-obj order_optimization
python test/TimeTest.py model2-normal order_optimization

# Run the greedy_structure_learning function
python test/TimeTest.py model2-normal greedy_structure_learning

# Run the tabu_structure_learning function
python test/TimeTest.py model2-normal tabu_structure_learning
