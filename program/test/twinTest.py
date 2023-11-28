from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.BuildModel import BuildModelFromConfig
from dl.DataLoader import make_dataloader

# get score
def getScore(config):
    modelName = config["modelName"]
    targetVar = config["targetVar"]
    controlVars = config["controlVars"]
    changeRates = config["changeRates"]

    # data
    dl = make_dataloader(None, None, None, None, modelName)
    dl.train_test_split()
    dl.pt_data = dl.test_data
    dataLen = len(dl.test_data)

    # list model configs
    modelPaths = os.listdir(os.path.join("data/modelData", modelName))
    scoreFilePath = os.path.join("data/modelData", modelName, "score.csv")
    if "score.csv" in modelPaths:
        return pd.read_csv(scoreFilePath)

    scoreList = []
    firstData = True
    # get score for each model
    for folder in modelPaths:
        folderPath = os.path.join("data/modelData", modelName, folder)
        if os.path.isdir(folderPath):
            models = os.listdir(folderPath)
            for modelNum in models:
                modelPath = os.path.join(folderPath, modelNum)
                print(modelPath)
                scores = [folder]
                try:
                    with open(modelPath, "r") as f:
                        modelConfig = json.load(f)
                except:
                    print("error in loading: ", modelPath)
                    continue
                model = BuildModelFromConfig(modelConfig)
                model.set_data_from_dataloader(dl)
                """
                where to get the score
                """
                scores.append(modelConfig.get("timeTaken", 0))
                scores.append(model.evaluate(targetVar, type="log_likelihood"))
                for controlVar in controlVars:
                    for changeRate in changeRates:
                        scores.append(model.evaluate(targetVar, controlVar=controlVar, changeRate=changeRate, type="elasticity", num_samples=dataLen))
                
                # make it dataframe
                scores = pd.DataFrame(np.array([scores]), columns=["model", "timeTaken", "log_likelihood"]+["elasticity_"+controlVar+"_"+str(changeRate) for controlVar in controlVars for changeRate in changeRates])
                with open(scoreFilePath, 'a') as file:
                    scores.to_csv(file, header=firstData, index=False)
                firstData = False

    return scoreList


# visualize
def visualize(modelName, scoreList, criterion="log_likelihood"):
    print("criterion: ", criterion)
    if criterion in ["log_likelihood", "timeTaken"]:
        metric = criterion
        scoreList = scoreList[["model", metric]]

        # Separate the true model and other models
        trueModelScore = scoreList[scoreList["model"] == "truth"][metric].values[0]
        scoreList = scoreList[scoreList["model"] != "truth"]

        # Aggregate by model type
        scoreListMean = scoreList.groupby("model").mean().reset_index()
        scoreListUpper = scoreList.groupby("model").quantile(0.95).reset_index()
        scoreListLower = scoreList.groupby("model").quantile(0.05).reset_index()

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot([0], [trueModelScore], color="red", label="True Model", marker="o", markersize=10)

        for i, modeltype in enumerate(scoreListMean["model"]):
            mean_value = scoreListMean[scoreListMean["model"] == modeltype][metric].values[0]
            upper_value = scoreListUpper[scoreListUpper["model"] == modeltype][metric].values[0]
            lower_value = scoreListLower[scoreListLower["model"] == modeltype][metric].values[0]
            plt.errorbar([i + 1], [mean_value], yerr=[[mean_value - lower_value], [upper_value - mean_value]], fmt='o', label=modeltype, capsize=5)

        plt.xticks(range(len(scoreListMean["model"]) + 1), ["True Model"] + list(scoreListMean["model"]), rotation=45)
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join("data/modelData", modelName, f"{metric}.png"))
        plt.close()

    if criterion == "elasticity":
        # Filter for elasticity columns
        elasticity_columns = [col for col in scoreList.columns if col.startswith("elasticity_")]
        controlVars = set('_'.join(col.split('_')[1:-1]) for col in elasticity_columns)
        changeRates = sorted(list(set(float(col.split('_')[-1]) for col in elasticity_columns)))

        for controlVar in controlVars:
            plt.figure(figsize=(10, 6))

            # Separate models
            models = scoreList["model"].unique()
            for model in models:
                model_data = scoreList[scoreList["model"] == model]

                # Prepare data for plotting for each model
                plot_data = []
                for changeRate in changeRates:
                    col_name = f"elasticity_{controlVar}_{changeRate}"
                    elasticity_values = model_data[col_name].dropna()
                    elasticity_values = elasticity_values[elasticity_values != 0]
                    mean_value = elasticity_values.mean()
                    upper_value = elasticity_values.quantile(0.95)
                    lower_value = elasticity_values.quantile(0.05)
                    plot_data.append((changeRate, mean_value, lower_value, upper_value))

                # Plotting for each model
                x, means, lowers, uppers = zip(*plot_data)
                plt.plot(x, means, label=f'{model} Mean Elasticity', marker='o')
                plt.fill_between(x, lowers, uppers, alpha=0.2, label=f'{model} 0.05-0.95 Quantile Range')

            plt.xlabel('Change Rate')
            plt.ylabel('Elasticity')
            plt.title(f'Elasticity of {controlVar} on {modelName}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join("data/modelData", modelName, f"elasticity_{controlVar}.png"))
            plt.close()



if __name__ == "__main__":
    # read config with argparse
    parser = argparse.ArgumentParser(description="Run a structure optimization experiment")

    # add the arguments
    parser.add_argument("--ConfigFile",
                        metavar="config_file",
                        type=str,
                        help="the path to config file")
    args = parser.parse_args()
    with open(args.ConfigFile, "r") as f:
        config = json.load(f)

    # get score
    scoreList = getScore(config)
    # visualize
    visualize(config["modelName"], scoreList, criterion="log_likelihood")
    visualize(config["modelName"], scoreList, criterion="elasticity")
    visualize(config["modelName"], scoreList, criterion="timeTaken")

