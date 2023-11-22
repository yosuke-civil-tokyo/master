from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import json
import pandas as pd
from model.BuildModel import BuildModelFromConfig

# get score and visualize
def getScore(config):
    modelName = config["modelName"]
    targetVar = config["targetVar"]
    controlVars = config["controlVar"]
    changeRates = config["changeRate"]
    # list model configs
    modelPaths = os.listdir(os.path.join("data", modelName))
    scoreList = []
    # get score for each model
    for modelPath in modelPaths:
        scores = [modelPath]
        with open(os.path.join("data", modelName, modelPath), "r") as f:
            config = json.load(f)
        model = BuildModelFromConfig(config)
        """
        where to get the score
        """
        scores.append(model.evaluate(targetVar, type="log_likelihood"))
        for controlVar in controlVars:
            for changeRate in changeRates:
                scores.append(model.evaluate(targetVar, controlVar=controlVar, changeRate=changeRate, type="elasticity"))
        scoreList.append(scores)

    # make it dataframe
    scoreList = pd.DataFrame(scoreList, columns=["model", "log_likelihood"]+["elasticity_"+controlVar+"_"+str(changeRate) for controlVar in controlVars for changeRate in changeRates])
    # save it
    scoreList.to_csv(os.path.join("data", modelName, "score.csv"), index=False)

    return scoreList

    
