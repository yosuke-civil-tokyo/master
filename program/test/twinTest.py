from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from model.BuildModel import BuildModelFromConfig

# get score
def getScore(config):
    modelName = config["modelName"]
    targetVar = config["targetVar"]
    controlVars = config["controlVars"]
    changeRates = config["changeRates"]
    # list model configs
    modelPaths = os.listdir(os.path.join("data/modelData", modelName))
    if "score.csv" in modelPaths:
        scoreList = pd.read_csv(os.path.join("data/modelData", modelName, "score.csv"))
        return scoreList

    modelPaths = [modelPath for modelPath in modelPaths if modelPath.endswith(".json")]
    modelPaths.sort()
    scoreList = []
    # get score for each model
    for modelPath in modelPaths:
        print(modelPath)
        scores = [modelPath]
        try:
            with open(os.path.join("data/modelData", modelName, modelPath), "r") as f:
                modelConfig = json.load(f)
        except:
            print("error in loading: ", modelPath)
            continue
        model = BuildModelFromConfig(modelConfig)
        scores.append(modelConfig.get("timeTaken", 0))
        model.generate(10000)
        """
        where to get the score
        """
        scores.append(model.evaluate(targetVar, type="log_likelihood"))
        for controlVar in controlVars:
            for changeRate in changeRates:
                scores.append(model.evaluate(targetVar, controlVar=controlVar, changeRate=changeRate, type="elasticity"))
        scoreList.append(scores)

    # make it dataframe
    scoreList = pd.DataFrame(scoreList, columns=["model", "timeTaken", "log_likelihood"]+["elasticity_"+controlVar+"_"+str(changeRate) for controlVar in controlVars for changeRate in changeRates])
    # save it
    scoreList.to_csv(os.path.join("data/modelData", modelName, "score.csv"), index=False)

    return scoreList


# visualize
def visualize(modelName, scoreList, criterion="log_likelihood"):
    if criterion=="log_likelihood":
        scoreList = scoreList[["model"  , "log_likelihood"]]
        # log_likelihood of true model
        trueModel = scoreList[scoreList["model"]=="truth.json"]["log_likelihood"].values[0]
        scoreList = scoreList[scoreList["model"]!="truth.json"]
        # log_likelihood of other models
        # aggregate with strings before "_"
        print(scoreList["model"].apply(lambda x: "_".join(x.split("_")[:-1])).values)
        scoreList["modeltype"] = scoreList["model"].apply(lambda x: "_".join(x.split("_")[:-1])).values
        scoreList = scoreList[["modeltype", "log_likelihood"]]
        # aggregate by modeltype
        scoreListMean = scoreList.groupby("modeltype").mean().reset_index()
        scoreListUpper = scoreList.groupby("modeltype").quantile(0.95).reset_index()
        scoreListLower = scoreList.groupby("modeltype").quantile(0.05).reset_index()
        # plot, dot of true model, and boxplot of other models
        plt.figure(figsize=(10, 6))
        plt.plot([0], [trueModel], color="red", label="true model", marker="o", markersize=10)
        # plt.boxplot(scoreListMean["log_likelihood"], positions=[1], widths=0.5, showmeans=True)
        plt.errorbar([1], scoreListMean["log_likelihood"].values[0], yerr=[[scoreListMean["log_likelihood"].values[0]-scoreListLower["log_likelihood"].values[0]], [scoreListUpper["log_likelihood"].values[0]-scoreListMean["log_likelihood"].values[0]]], fmt='o', color='black', capsize=5)
        plt.xticks([0, 1], ["true model", "pred model"])
        plt.ylabel("log_likelihood")
        plt.legend()
        plt.savefig(os.path.join("data/modelData", modelName, "log_likelihood.png"))

    elif criterion=="elasticity":
        # Filter for elasticity columns
        elasticity_columns = [col for col in scoreList.columns if col.startswith("elasticity_")]
        controlVars = set('_'.join(col.split('_')[1:-1]) for col in elasticity_columns)
        changeRates = sorted(list(set(float(col.split('_')[-1]) for col in elasticity_columns)))

        for controlVar in controlVars:
            # Prepare data for plotting
            plot_data = []
            for changeRate in changeRates:
                col_name = f"elasticity_{controlVar}_{changeRate}"
                elasticity_values = scoreList[col_name].dropna()
                elasticity_values = elasticity_values[elasticity_values != 0]
                mean_value = elasticity_values.mean()
                upper_value = elasticity_values.quantile(0.95)
                lower_value = elasticity_values.quantile(0.05)
                plot_data.append((changeRate, mean_value, lower_value, upper_value))

            # Plotting
            plt.figure(figsize=(10, 6))
            x, means, lowers, uppers = zip(*plot_data)
            plt.plot(x, means, label='Mean Elasticity', marker='o')
            plt.fill_between(x, lowers, uppers, alpha=0.2, label='0.05-0.95 Quantile Range')
            plt.xlabel('Change Rate')
            plt.ylabel('Elasticity')
            plt.title(f'Elasticity of {controlVar} on {modelName}')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join("data/modelData", modelName, f"elasticity_{controlVar}.png"))
            plt.close()

    # the same as log_likelihood
    elif criterion=="timeTaken":
        scoreList = scoreList[["model"  , "timeTaken"]]
        # timeTaken of true model
        trueModel = scoreList[scoreList["model"]=="truth.json"]["timeTaken"].values[0]
        scoreList = scoreList[scoreList["model"]!="truth.json"]
        # timeTaken of other models
        # aggregate with strings before "_"
        scoreList["modeltype"] = scoreList["model"].apply(lambda x: "_".join(x.split("_")[:-1])).values
        scoreList = scoreList[["modeltype", "timeTaken"]]
        # aggregate by modeltype
        scoreListMean = scoreList.groupby("modeltype").mean().reset_index()
        scoreListUpper = scoreList.groupby("modeltype").quantile(0.95).reset_index()
        scoreListLower = scoreList.groupby("modeltype").quantile(0.05).reset_index()
        # plot, dot of true model, and boxplot of other models
        plt.figure(figsize=(10, 6))
        plt.plot([0], [trueModel], color="red", label="true model", marker="o", markersize=10)
        # plt.boxplot(scoreListMean["timeTaken"], positions=[1], widths=0.5, showmeans=True)
        plt.errorbar([1], scoreListMean["timeTaken"].values[0], yerr=[[scoreListMean["timeTaken"].values[0]-scoreListLower["timeTaken"].values[0]], [scoreListUpper["timeTaken"].values[0]-scoreListMean["timeTaken"].values[0]]], fmt='o', color='black', capsize=5)
        plt.xticks([0, 1], ["true model", "pred model"])
        plt.ylabel("timeTaken")
        plt.legend()
        plt.savefig(os.path.join("data/modelData", modelName, "timeTaken.png"))



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

