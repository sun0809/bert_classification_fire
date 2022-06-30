import os
import pandas as pd
import json

PATH = 'C:\\test_github\\bert_classification\\04.07\\'
folder_name = 'docc_Folder\\'
data_folder = f'{PATH}{folder_name}'
file_list = os.listdir(data_folder)
def jsonToTxt():
    for i in file_list:
        if i[-4:] == 'json':
            name = i[-4:]
            with open(f'{data_folder}{i}', 'r', encoding = 'utf-8') as f:
                json_data = json.load(f)
        
                dt3 = pd.DataFrame(json_data)
                dt4 = dt3[['label','data']]
                dt4.loc[:,'label'] = dt4.loc[:,'label'].apply(lambda x : x[0])
                dt4.to_csv(f'{data_folder}{name}.txt', sep = '\t', encoding = 'utf-8', header = None, index=False)
                print(f'----------------{data_folder}{name}txt 완료---------------------')


jsonToTxt()