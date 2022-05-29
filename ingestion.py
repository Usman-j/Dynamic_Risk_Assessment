import pandas as pd
import os
import json
from datetime import datetime



#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
dataset_csv_path = os.path.join(os.getcwd(),config['output_folder_path'])



#############Function for data ingestion
def merge_multiple_dataframe():
    '''
    Checks for datasets, compiles them together, writes to an output file and saves record of ingested data.
    '''
    df_final = pd.DataFrame()
    record_file = open(dataset_csv_path+'/ingestedfiles.txt', 'w')
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    record_file.write('Ingestion Datetime: '+ dt_string+ '\n')
    record_file.write('Data Folder: '+ input_folder_path+ '\n')
    filenames = os.listdir(os.getcwd()+'/'+input_folder_path)
    for file in filenames:
        if file[-4:] == '.csv':
            record_file.write(file+'\n')
            temp_df = pd.read_csv(os.getcwd()+'/'+input_folder_path+'/'+file)
            df_final = df_final.append(temp_df)
    record_file.close()
    df_final = df_final.drop_duplicates()
    df_final.to_csv(dataset_csv_path+'/finaldata.csv', index=False)

if __name__ == '__main__':
    merge_multiple_dataframe()
