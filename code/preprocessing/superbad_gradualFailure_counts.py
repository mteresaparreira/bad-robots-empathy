"""
Calculate the total number of Sudden and Gradual Videos that are present per each class of videos
"""

import numpy as np
import pandas as pd

dataset_path = '../../data/survey_log/New_Stimulus_Dataset_Information.xlsx'
df = pd.read_excel(dataset_path)

#### Check how many are gradual and sudden failures

allSuddenGradual_count_dict = {
    'failure, sudden': 0,
    'failure, gradual': 0,
    'control, sudden': 0,
    'control, gradual': 0,
}

for index, row in df.iterrows():
    if row['Failure/Control Video'] == 'Failure' and row['Sudden/Gradual'] == 'Sudden':
        allSuddenGradual_count_dict['failure, sudden'] += 1
    elif row['Failure/Control Video'] == 'Failure' and row['Sudden/Gradual'] == 'Gradual':
        allSuddenGradual_count_dict['failure, gradual'] += 1
    if row['Failure/Control Video'] == 'Control' and row['Sudden/Gradual'] == 'Sudden':
        allSuddenGradual_count_dict['control, sudden'] += 1
    elif row['Failure/Control Video'] == 'Control' and row['Sudden/Gradual'] == 'Gradual':
        allSuddenGradual_count_dict['control, gradual'] += 1

total_sudden_count = allSuddenGradual_count_dict['failure, sudden'] + allSuddenGradual_count_dict['control, sudden']
total_failure_count = allSuddenGradual_count_dict['failure, gradual'] + allSuddenGradual_count_dict['control, gradual']

print('------------------------------')
print(f'All Stimulus Video:\nTotal Sudden Count: {total_sudden_count} & Failure Count: {total_failure_count}')
print(allSuddenGradual_count_dict)

#### For the final shortlisted videos, what are the counts

finalSuddenGradual_count_dict = {
    'failure, sudden': 0,
    'failure, gradual': 0,
    'control, sudden': 0,
    'control, gradual': 0,
}

final_df = df[df['Final Count'] != 'NoInclude']

for index, row in final_df.iterrows():
    if row['Failure/Control Video'] == 'Failure' and row['Sudden/Gradual'] == 'Sudden':
        finalSuddenGradual_count_dict['failure, sudden'] += 1
    elif row['Failure/Control Video'] == 'Failure' and row['Sudden/Gradual'] == 'Gradual':
        finalSuddenGradual_count_dict['failure, gradual'] += 1
    if row['Failure/Control Video'] == 'Control' and row['Sudden/Gradual'] == 'Sudden':
        finalSuddenGradual_count_dict['control, sudden'] += 1
    elif row['Failure/Control Video'] == 'Control' and row['Sudden/Gradual'] == 'Gradual':
        finalSuddenGradual_count_dict['control, gradual'] += 1

total_sudden_count = finalSuddenGradual_count_dict['failure, sudden'] + finalSuddenGradual_count_dict['control, sudden']
total_failure_count = finalSuddenGradual_count_dict['failure, gradual'] + finalSuddenGradual_count_dict['control, gradual']

print('------------------------------')
print(f'Final Stimulus Videos:\nTotal Sudden Count: {total_sudden_count} & Failure Count: {total_failure_count}')
print(finalSuddenGradual_count_dict)
print('------------------------------')