"""
This Python file is example of how your `pred.py` script should
take in the input, and produce an output. Your `pred.py` script can
use different methods to process the input data, but the format
of the input it takes and the output your script produces should be
the same.

Usage:
    example_pred.py <test_data.csv>
"""

# basic python imports are permitted
import sys
import csv
import random
import pandas as pd
import Model_MLP
import Model_NaiveBayes
import Model_LogisticRegression
# numpy and pandas are also permitted


class combineModel:
    def __init__(self, modelList, accuracyList) -> None:
        self.modelList = modelList
        self.accuracyList = accuracyList

    # Based on the accuracy, if there are two value that has the same amount of output,
    # choose the one from highest average accuracy
    def predict(self, x, x2):
        t = []
        for data in range(len(x)):
            output = {1: [], 2: [], 3: []}
            for i in range(len(self.modelList)):
                if(isinstance(self.modelList[i], Model_LogisticRegression.LogisticReg)):
                    output[int(self.modelList[i].predict(x2[data]))].append(i)
                else:
                    output[self.modelList[i].predict(x[data])[0]].append(i)
            output2 = {1: len(output[1]), 2: len(output[2]), 3: len(output[3])}
            # generate a list of output with the highest same amount of model prediction
            maxKeys = [key for key, value in output2.items() if value ==
                    max(output2.values())]
            # if theres only one prediction that most model predicts
            y = -1
            if len(maxKeys) == 1:
                y = maxKeys[0]
            else:
                maxAccuracy = 0
                for i in range(len(maxKeys)):
                    currModelList = output[maxKeys[i]]
                    currAccuracy = 0
                    # take the index from the model that predicts maxKeys[i] and find its correspond accuracy
                    for m in currModelList:
                        currAccuracy += accuracyList[m]
                    currAccuracy = currAccuracy / len(currModelList)
                    if currAccuracy > maxAccuracy:
                        maxAccuracy == currAccuracy
                        y = maxKeys[i]
            t.append(y)
        # print the prediction to stdout

        return t


if __name__ == "__main__":
    # check if the argument <test_data.csv> is provided
    if len(sys.argv) < 2:
        print("""
Usage:
    python example_pred.py <test_data.csv>

As a first example, try running `python example_pred.py example_test_set.csv`
""")
        exit()
    MLPmodel = Model_MLP.model11
    NaiveBayesmodel = Model_NaiveBayes.model
    # LogisticReg = Model_LogisticRegression.model
    modelList = [MLPmodel, NaiveBayesmodel]
    accuracyList = [0.7319587628865979, 0.6857938144329897]
    combineModel = combineModel(modelList, accuracyList)
    # store the name of the file containing the test data
    filename = sys.argv[-1]

    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    df = pd.read_csv(filename)
    # df2 = Model_LogisticRegression.process_log(df.copy())
    df = Model_MLP.process_data(df)
    df.to_csv("example2.csv")
    if ("label" in df.keys()):
        x = df.drop(["label", "user_id"], axis=1).values
        # x2 = df2.drop(["label", "user_id"], axis=1).values
        y = df["label"].values
    else:
        x   = df.drop(["user_id"], axis=1).values
        # x2   = df2.drop(["user_id"], axis=1).values
        # obtain a prediction for this test example
    t = combineModel.predict(x, None)
    for pred in t:
        print(pred)
    # compute accuracy
    # n = len(x)
    # c = 0
    # for i in range(n):  
    #     if y[i] == t[i]:
    #         c += 1
    # print("accuracy: ", c/n)
