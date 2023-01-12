import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_model_path']) 
final_data_path = os.path.join(config['output_folder_path']) 

##############Function for reporting
def score_model():
    data = open(f'./{final_data_path}/finaldata.csv', 'r')
    predicted = model_predictions(data)
    predicted = predicted.tolist()
    df = pd.read_csv(f'./{final_data_path}/finaldata.csv')
    actual = df['exited'].to_list()
    plot = metrics.confusion_matrix(actual, predicted)
    plt.imshow(plot, cmap='binary', interpolation='None')
    plt.savefig(f'./{dataset_csv_path}/confusionmatrix2.png')
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

if __name__ == '__main__':
    score_model()
