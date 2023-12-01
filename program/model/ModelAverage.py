import time
import numpy as np

from model.DeepGen import DeepGenerativeModel

# average function for structure
# the one calculate each arc's reliability 
# and add to the model if it is higher than threshold
def thresAverage(configs, beta=0.7):
    startTime = time.time()
    variableNames = list(configs[0].get("variables").keys())
    variableNums = {variableNames[i]: i for i in range(len(variableNames))}
    countTable = np.zeros((len(variableNames), len(variableNames)))

    for config in configs:
        variables = config.get("variables")
        for child, info in variables.items():
            for parent in info.get("parents"):
                countTable[variableNums[child], variableNums[parent]] += 1
    
    countTable = countTable / len(configs)
    countTable = countTable > beta
    aveConfig = configs[0]
    aveConfig = setArcs(aveConfig, countTable, variableNames)
    return [aveConfig], time.time()-startTime

def bestChoice(configs):
    startTime = time.time()
    scores = [config.get("score") for config in configs]
    bestIndex = scores.index(max(scores))
    return [configs[bestIndex]], time.time()-startTime

def deepAverage(configs, num_samples=100, sample_per_model=10):
    startTime = time.time()
    model = DeepGenerativeModel()
    model.train(configs)
    model_samples = model.sample(num_samples//sample_per_model)
    aveConfig = configs[0]
    varaibleNames = list(aveConfig.get("variables").keys())
    configs = [aveConfig for _ in range(len(model_samples))]
    for i in range(len(model_samples)):
        configs[i] = setArcs(configs[i], model_samples[i], varaibleNames)

    return configs, time.time()-startTime

def setArcs(config, countTable, variableNames):
    variableNums = {variableNames[i]: i for i in range(len(variableNames))}
    variables = config.get("variables")
    for child, info in variables.items():
        for parent in info.get("parents"):
            if countTable[variableNums[child], variableNums[parent]]:
                variables[child]["parents"].append(parent)
            else:
                variables[child]["parents"].remove(parent)
    config["variables"] = variables
    return config