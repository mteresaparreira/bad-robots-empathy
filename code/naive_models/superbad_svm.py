import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def trainModel(df, param_grid, results_directory):
    # # Extract features and labels

    # # for naive & naive_n datasets
    # features = df.iloc[:, 3:]
    # target_class = df['class'].values

    # # for full & full_n datasets
    features = df.iloc[:, 4:]
    target_class = df.iloc[:, 2].values
    target_class = target_class.astype('int')

    ## Split the data
    X_train, X_test, y_train, y_test = train_test_split(features, target_class, test_size = 0.20, random_state = 42)

    # SVM without Grid Search
    print('hello')
    classifier = svm.SVC(decision_function_shape='ovr', random_state=42)
    classifier.fit(X_train, y_train)

    y_predict = classifier.predict(X_test)


    # print(classification_report(y_test, y_predict))

    # Calculate the confusion matrix without training the model through gridsearch
    conf_matrix = confusion_matrix(y_test, y_predict)

    # Get the class labels (assuming y_true and grid_predictions are integer class labels)
    class_labels = ['Control', 'Failure_Human', 'Failure_Robot']

    # Plot the confusion matrix using seaborn and matplotlib
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save the confusion matrix as an image
    confusion_matrix_path = results_directory + 'svm_confusion_matrix_beforeGridSearch.png'
    plt.savefig(confusion_matrix_path)
    plt.clf()

    with open(f'{results_directory}/superBAD_svm.txt', 'a') as output_file:
        output_file.write('---------------------------------------------------------------------')
        output_file.write('\n' + 'Begin Model: SVM without GridSearch' + '\n')
        output_file.write('Dataset: allParticipants_5fps, full_n' + '\n')
        output_file.write(f"---------CONFUSION MATRIX STORED AT PATH : {confusion_matrix_path}-----------" + '\n')
        output_file.write("---------CLASSIFICATION REPORT-----------" + '\n')
        output_file.write(classification_report(y_test, y_predict))
        output_file.write("---------END---------" + '\n')
        output_file.write('---------------------------------------------------------------------')

    grid_search = GridSearchCV(svm.SVC(decision_function_shape='ovr', random_state = 42), param_grid=param_grid, refit= True, verbose = 10, error_score = 999999999)
    grid_search.fit(X_train, y_train)
    
    # Extract the cv_results_ attribute from GridSearchCV
    cv_results = grid_search.cv_results_

    # Convert cv_results into a DataFrame
    df = pd.DataFrame(cv_results)
    df.to_csv(f'{results_directory}/svm_grid_search_cross_validation_results.csv')

    # print best parameter after tuning
    # print("Best Parameters: ", grid_search.best_params_)
    
    # print how our model looks after hyper-parameter tuning
    # print(grid_search.best_estimator_)

    grid_predictions = grid_search.predict(X_test)
    
    # print classification report
    # print(classification_report(y_test, grid_predictions))

    # Calculate the confusion matrix after training models through gridsearch
    conf_matrix = confusion_matrix(y_test, grid_predictions)

    # Get the class labels (assuming y_true and grid_predictions are integer class labels)
    class_labels = ['Control', 'Failure_Human', 'Failure_Robot']

    # Plot the confusion matrix using seaborn and matplotlib
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    # Save the confusion matrix as an image
    confusion_matrix_path = results_directory + 'svm_confusion_matrix_afterGridSearch.png'
    plt.savefig(confusion_matrix_path)
    plt.clf()

    with open(f'{results_directory}/superBAD_svm.txt', 'a') as output_file:
        output_file.write('---------------------------------------------------------------------' + '\n')
        output_file.write('Begin Model: SVM with GridSearch' + '\n')
        output_file.write('Dataset: allParticipants_5fps, full_n' + '\n')
        output_file.write('---------------------------------------'+ '\n')
        output_file.write('BEST PARAMETERS: ' + '\n')
        output_file.write(str(grid_search.best_params_)+ '\n')
        output_file.write('---------------------------------------'+ '\n')
        output_file.write('BEST ESTIMATOR: ' + '\n')
        output_file.write(str(grid_search.best_estimator_)+ '\n')
        output_file.write('---------------------------------------'+ '\n')
        output_file.write('BEST SCORE: ' + '\n')
        output_file.write(str(grid_search.best_score_)+ '\n')
        output_file.write('---------------------------------------'+ '\n')
        output_file.write('BEST CANDIDATE PARAMETER INDEX IN THE CV RESULTS DataFrame: ' + '\n')
        output_file.write(str(grid_search.best_index_)+ '\n')
        output_file.write('---------------------------------------'+ '\n')
        output_file.write(f"---------CONFUSION MATRIX STORED AT PATH : {confusion_matrix_path}-----------" + '\n')
        output_file.write("---------CLASSIFICATION REPORT-----------" + '\n')
        output_file.write(classification_report(y_test, grid_predictions))
        output_file.write("---------END---------" + '\n')
        output_file.write('---------------------------------------------------------------------' + '\n')

def main():

    df = pd.read_csv('../../data/allParticipant_data/allParticipants_5fps_downsampled_preprocessed_norm.csv')
    
    # Specify the output directory
    now = datetime.datetime.now()
    results_directory = '../../results/' + 'SVM_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'
        
    # Create 'results_directory' if it doesn't exist
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # SVM with GridSearch
    gamma_exp = [-15,-13,-11,-9,-7,-5,-3,-1,1,3]
    c_exp = [-5,-3,-1,1,3,5,7,9,11,13,15]

    gamma_list = []
    c_list = []

    for i in gamma_exp:
        gamma_list.append(2**i)

    for i in c_exp:
        c_list.append(2**i)

    param_grid = {
        'kernel': ['rbf', 'sigmoid', 'poly'],
        'C': c_list,
        'gamma': gamma_list
    }

    trainModel(df, param_grid, results_directory)

if __name__ == "__main__":
    main()