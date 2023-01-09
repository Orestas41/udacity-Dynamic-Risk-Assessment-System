import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']
file_record=open(f'./{output_folder_path}/ingestedfiles.txt','a')

#############Function for data ingestion
def merge_multiple_dataframe():
    data = pd.DataFrame()
    datasets = os.listdir(os.getcwd()+'/'+input_folder_path)
    for each_dataset in datasets:
        file_record.write(str(each_dataset)+'\n')
        df = pd.read_csv(os.getcwd()+'/'+input_folder_path+'/'+each_dataset)
        data = data.append(df)
    result=data.drop_duplicates()
    result.to_csv(f'./{output_folder_path}/finaldata.csv', index=False)

    #check for datasets, compile them together, and write to an output file

if __name__ == '__main__':
    merge_multiple_dataframe()