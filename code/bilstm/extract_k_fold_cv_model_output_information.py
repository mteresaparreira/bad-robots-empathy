import re
import os
import ast
import statistics
import numpy as np
import pandas as pd

def parseResultsFile(file_path):
    with open(file_path, 'r') as results_file:
        results_content = results_file.read()

    cleaned_content = [line for line in results_content.split('\n')]

    search_combinations = []
    current_instance = []

    for line in cleaned_content:
        if re.search(r'FOLD :', line):
            if current_instance:
                search_combinations.append(current_instance)
            current_instance = [line]
        elif re.search(r'END', line):
            if current_instance:
                current_instance.append(line)
                search_combinations.append(current_instance)
            current_instance = []
        elif current_instance:
            current_instance.append(line)
    
    ## Display each fold
    # for instance in search_combinations:
    #     for line in instance:
    #         print(line)
    #     break

    search_info = []

    for search in search_combinations:
        fold_num = 0
        classification_report = ''
        capture_classification = False
        confusion_matrix = ''

        for line in search:
            # print(line)
            fold_match = re.search(r'FOLD = (\d+)', line)
            if fold_match:
                fold_num = ast.literal_eval(fold_match.group(1))

            class_report_start = re.search(r'------------ CLASSIFICATION REPORT ------------', line)
            if class_report_start:
                capture_classification = True  # Start capturing content
                continue  # Skip this line as it's just a marker
            elif capture_classification and re.search(r'------------ CONFUSION MATRIX ------------', line):
                capture_classification = False  # Stop capturing on "Confusion Matrix" marker
            elif capture_classification:
                if line.strip() != '':
                    classification_report += line + '\n'

            # Capture the lines within the "CONFUSION MATRIX" section
            if re.search(r'Confusion matrix saved for', line):
                confusion_matrix = line.split('Confusion matrix saved for ')[-1]  # Extract file name
                break  # Stop capturing after saving the confusion matrix filename
        # print(fold_num)
        ## Define the variables to be extracted in the classification report
        capture_classification_row = False
        test_accuracy = 0
        precision_values = {}
        recall_values = {}
        f1_scores = {}
        macro_avgs = {}
        weighted_avgs = {}

        # RE pattern for obtaining classification reports values
        test_accuracy_pattern = r'\s+(\d+\.\d+)\s+'
        class_values_pattern = r'^\s*(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+'
        macro_avg_pattern = r'\s*macro avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'
        weighted_avg_pattern = r'\s*weighted avg\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'

        # Once we have the classification report, let us extract all the other information in it
        for row in classification_report.split('\n'):
            row_headers_start_match = re.search("precision    recall  f1-score   support", row)
            if row_headers_start_match:
                capture_classification_row = True
                continue
            elif capture_classification_row and re.search('accuracy', row):
                accuracy_match = re.search(test_accuracy_pattern, row)
                if accuracy_match:
                    test_accuracy = ast.literal_eval(accuracy_match.group(1))
            elif capture_classification_row and re.search('macro avg', row):
                macro_avg_match = re.search(macro_avg_pattern, row)
                if macro_avg_match:
                    macro_avgs['precision'] = ast.literal_eval(macro_avg_match.group(1))
                    macro_avgs['recall'] = ast.literal_eval(macro_avg_match.group(2))
                    macro_avgs['f1_score'] = ast.literal_eval(macro_avg_match.group(3))
                    
            elif capture_classification_row and re.search('weighted avg', row):
                weighted_avg_match = re.search(weighted_avg_pattern, row)
                if weighted_avg_match:
                    weighted_avgs['precision'] = ast.literal_eval(weighted_avg_match.group(1))
                    weighted_avgs['recall'] = ast.literal_eval(weighted_avg_match.group(2))
                    weighted_avgs['f1_score'] = ast.literal_eval(weighted_avg_match.group(3))
                    capture_classification_row = False
                    # break
            elif capture_classification_row and re.search(r'\d+', row):
                class_values_match = re.search(class_values_pattern, row)
                if class_values_match:
                    class_label = ast.literal_eval(class_values_match.group(1))
                    precision_value = ast.literal_eval(class_values_match.group(2))
                    recall_value = ast.literal_eval(class_values_match.group(3))
                    f1_score = ast.literal_eval(class_values_match.group(4))
                    
                    precision_values[class_label] = precision_value
                    recall_values[class_label] = recall_value
                    f1_scores[class_label] = f1_score
        
        # print(f'Classification Report:\n{classification_report}')
        # print(f'Confusion Matrix: {confusion_matrix}')
        # print(f'Precison Values : {precision_values}')
        # print(f'Recall Values: {recall_values}')
        # print(f'F1-Scores: {f1_scores}')
        # print(f'Macro Averages: {macro_avgs}')
        # print(f'Weighted Averages: {weighted_avgs}')

        search_info.append({
            'fold_num' : fold_num,
            'accuracy' : test_accuracy,
            'precision_values': precision_values,
            'recall_values': recall_values,
            'f1_scores': f1_scores,
            'macro_averages': macro_avgs,
            'weighted_averages': weighted_avgs,
            'classification_report': classification_report,
            'confusion_matrix_path': confusion_matrix
        })
    return search_info

# Define a function to extract a specific key from each dictionary
def extract_key(dictionary_list, key):
    return [d[key] for d in dictionary_list]

def extractInformation(file_path, output_path):

    search_info = parseResultsFile(file_path)

    columns = ['fold_num', 'accuracy', 'precision_values', 'recall_values', 'f1_scores', 'macro_averages', 'weighted_averages', 'classification_report','confusion_matrix_path']

    df = pd.DataFrame(search_info, columns=columns)
    df.to_csv(output_path, index=False)

    accuracy_mean = df['accuracy'].mean()
    accuracy_std = df['accuracy'].std()
    
    print(f"Accuracy Mean: {accuracy_mean:.4f}")
    print(f"Accuracy Std: {accuracy_std:.4f}")
    ### Calculate the mean and stds

    # Macro Averages means & stds
    macro_averages_vals = df['macro_averages']
    # Initialize lists to store precision, recall, and f1_score values
    precision_values = []
    recall_values = []
    f1_score_values = []

    # Extract values from each dictionary in the 'macro_averages' column
    for row in macro_averages_vals:
        precision_values.append(row['precision'])
        recall_values.append(row['recall'])
        f1_score_values.append(row['f1_score'])

    # Calculate means and standard deviations
    precision_mean = statistics.mean(precision_values)
    precision_std = statistics.stdev(precision_values)

    recall_mean = statistics.mean(recall_values)
    recall_std = statistics.stdev(recall_values)

    f1_score_mean = statistics.mean(f1_score_values)
    f1_score_std = statistics.stdev(f1_score_values)

    print(f"Macro Averages Precision Mean: {precision_mean:.4f}")
    print(f"Macro Averages Precision Std: {precision_std:.4f}")

    print(f"Macro Averages Recall Mean: {recall_mean:.4f}")
    print(f"Macro Averages Recall Std: {recall_std:.4f}")

    print(f"Macro Averages F1 Score Mean: {f1_score_mean:.4f}")
    print(f"Macro Averages F1 Score Std: {f1_score_std:.4f}")
    
    # Weighted Averages means & stds
    weighted_averages_vals = df['weighted_averages']
    # Initialize lists to store precision, recall, and f1_score values
    precision_values = []
    recall_values = []
    f1_score_values = []

    # Extract values from each dictionary in the 'macro_averages' column
    for row in weighted_averages_vals:
        precision_values.append(row['precision'])
        recall_values.append(row['recall'])
        f1_score_values.append(row['f1_score'])

    # Calculate means and standard deviations
    precision_mean = statistics.mean(precision_values)
    precision_std = statistics.stdev(precision_values)

    recall_mean = statistics.mean(recall_values)
    recall_std = statistics.stdev(recall_values)

    f1_score_mean = statistics.mean(f1_score_values)
    f1_score_std = statistics.stdev(f1_score_values)

    print(f"Weighted Averages Precision Mean: {precision_mean:.4f}")
    print(f"Weighted Averages Precision Std: {precision_std:.4f}")

    print(f"Weighted Averages Recall Mean: {recall_mean:.4f}")
    print(f"Weighted Averages Recall Std: {recall_std:.4f}")

    print(f"Weighted Averages F1 Score Mean: {f1_score_mean:.4f}")
    print(f"Weighted Averages F1 Score Std: {f1_score_std:.4f}")

def main():

    ### Change the paths to the required model
    
    file_path = '../../model_results/bilstm_results/k_fold_cv/k_fold_cv_overlapping_data_bilstm_final_2023-09-28_19-54-34/k_fold_cv_overlapping_data_results_BiLSTM_final.txt'
    output_path = '../../model_results/bilstm_results/k_fold_cv/k_fold_cv_overlapping_data_bilstm_final_2023-09-28_19-54-34/k_fold_cv_overlapping_data_results_BiLSTM_final.csv'

    extractInformation(file_path, output_path)


if __name__ == "__main__":
    main()
