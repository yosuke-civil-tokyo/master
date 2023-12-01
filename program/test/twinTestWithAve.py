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
from model.ModelAverage import thresAverage, bestChoice, deepAverage
from dl.DataLoader import make_dataloader

# get score
def getScore(config):
    modelName = config["modelName"]
    targetVar = config["targetVar"]
    controlVars = config["controlVars"]
    changeRates = config["changeRates"]
    averageMethod = config["averageMethod"]

    # data
    dl = make_dataloader(None, None, None, None, modelName)
    dl.train_test_split()
    dl.pt_data = dl.test_data
    dataLen = len(dl.test_data)

    # list model configs
    modelPaths = os.listdir(os.path.join("data/modelData", modelName))
    scoreFilePath = os.path.join("data/modelData", modelName, "scoreAve.csv")
    if "scoreAve.csv" in modelPaths:
        return pd.read_csv(scoreFilePath)

    scoreList = []
    firstData = True

    with open(os.path.join("data/modelData", modelName, "truth", "truth.json"), "r") as f:
        truthConfig = json.load(f)
    # get score for each model
    for folder in modelPaths:
        folderPath = os.path.join("data/modelData", modelName, folder)
        if os.path.isdir(folderPath):
            models = os.listdir(folderPath)
        # read models
        modelConfigs = []
        for modelNum in models:
            modelPath = os.path.join(folderPath, modelNum)
            try:
                with open(modelPath, "r") as f:
                    modelConfig = json.load(f)
                    
            except:
                print("error in loading: ", modelPath)
                continue
            modelConfigs.append(modelConfig)
        # average
        if averageMethod == "thres":
            aveConfigs, calTime = thresAverage(modelConfigs)
        elif averageMethod == "best":
            aveConfigs, calTime = bestChoice(modelConfigs)
        elif averageMethod == "deep":
            aveConfigs, calTime = deepAverage(modelConfigs, num_samples=dataLen, sample_per_model=10, modelName=folder)
        else:
            print("average method not found")
            return
        
        # get one score for the group of Configs
        # merge multiple result to get one score
        score = [folder, calTime]
        aveScore = []
        # BIC, log_likelihood, elasticity
        i = 0
        for aveConfig in aveConfigs:
            dataRange = (i*dataLen//len(aveConfigs), (i+1)*dataLen//len(aveConfigs))
            eachScore = []
            model = BuildModelFromConfig(aveConfig)
            model.set_data_from_dataloader(dl, dataRange=dataRange)

            # scores
            eachScores.append(edgeDetectAccuracy(modelConfig, truthConfig))
            # add loglikelihood and BIC for every variable
            LLBICcol = []
            for variable in truthConfig.get("variables").keys():
                LLBICcol.append(variable+"_log_likelihood")
                LLBICcol.append(variable+"_BIC")
                eachScores.append(calculate_log_likelihood(model.find_variable(variable)))
                eachScores.append(calculate_BIC(model.find_variable(variable)))
            if folder == "truth":
                tryTime = 10
            else:
                tryTime = 1
            for controlVar in controlVars:
                for changeRate in changeRates:
                    eachScores.append(model.evaluate(targetVar, controlVar=controlVar, changeRate=changeRate, type="elasticity", num_samples=dataLen, tryTime=tryTime))
            i += 1
            aveScore.append(eachScores)
        aveScore = np.array(aveScore)
        score.append(np.mean(aveScore, axis=0))

        # make it dataframe
        score = pd.DataFrame(np.array([score]), columns=["model", "timeTaken", "log_likelihood", "BIC"]+["elasticity_"+controlVar+"_"+str(changeRate) for controlVar in controlVars for changeRate in changeRates])
        with open(scoreFilePath, 'a') as file:
            score.to_csv(file, header=firstData, index=False)
        firstData = False

    return pd.read_csv(scoreFilePath)


# visualize
def visualize(modelName, scoreList, criterion="log_likelihood"):
    print("criterion: ", criterion)
    if criterion in ["log_likelihood", "BIC", "timeTaken"]:
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
    visualize(config["modelName"], scoreList, criterion="BIC")
    visualize(config["modelName"], scoreList, criterion="elasticity")
    visualize(config["modelName"], scoreList, criterion="timeTaken")

