import numpy as np

# average function for structure
# the one calculate each arc's reliability 
# and add to the model if it is higher than threshold
def thresAverage(configs, beta=0.7):
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
    return [aveConfig]

def bestChoice(configs):
    scores = [config.get("score") for config in configs]
    bestIndex = scores.index(max(scores))
    return [configs[bestIndex]]

def deepAverage(configs, num_samples=100, sample_per_model=10):
    model = DeepGenerativeModel()
    model.train(configs)
    samples = model.sample(num_samples//sample_per_model)
    aveConfig = configs[0]
    varaiblenames = list(aveConfig.get("variables").keys())
    configs = [aveConfig for _ in range(len(samples))]
    for i in range(len(samples)):
        configs[i] = setArcs(configs[i], samples[i], variableNames)

    return configs

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