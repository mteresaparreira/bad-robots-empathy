import os
import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
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

    # Split the data

    X_train, X_test, y_train, y_test = train_test_split(features, target_class, test_size = 0.20, random_state = 42)

    # RandomForests without Grid Search
    classifier = RandomForestClassifier(random_state = 42)
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
    confusion_matrix_path = results_directory + 'randomForest_confusion_matrix_beforeGridSearch.png'
    plt.savefig(confusion_matrix_path)
    plt.clf()

    with open(f'{results_directory}/superBAD_randomForests.txt', 'a') as output_file:
        output_file.write('---------------------------------------------------------------------')
        output_file.write('\n' + 'Begin Model: RandomForests without GridSearch' + '\n')
        output_file.write('Dataset: allParticipants_5fps, full_n' + '\n')
        output_file.write(f"---------CONFUSION MATRIX STORED AT PATH : {confusion_matrix_path}-----------" + '\n')
        output_file.write("---------CLASSIFICATION REPORT-----------" + '\n')
        output_file.write(classification_report(y_test, y_predict))
        output_file.write("---------END---------" + '\n')
        output_file.write('---------------------------------------------------------------------')

    grid_search = GridSearchCV(RandomForestClassifier(random_state = 42), param_grid=param_grid, refit= True, verbose = 4, error_score = 999999999)
    grid_search.fit(X_train, y_train)
    
    # Extract the cv_results_ attribute from GridSearchCV
    cv_results = grid_search.cv_results_

    # Convert cv_results into a DataFrame
    df = pd.DataFrame(cv_results)
    df.to_csv(f'{results_directory}/random_forests_grid_search_cross_validation_results.csv')

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
    confusion_matrix_path = results_directory + 'randomForest_confusion_matrix_afterGridSearch.png'
    plt.savefig(confusion_matrix_path)
    plt.clf()

    with open(f'{results_directory}/superBAD_randomForests.txt', 'a') as output_file:
        output_file.write('---------------------------------------------------------------------' + '\n')
        output_file.write('Begin Model: RandomForests with GridSearch' + '\n')
        output_file.write('Dataset: allParticipants_5fps, full_n' + '\n')
        output_file.write('---------------------------------------')
        output_file.write('BEST PARAMETERS: ' + '\n')
        output_file.write(grid_search.best_params_)
        output_file.write('BEST ESTIMATOR: ' + '\n')
        output_file.write(grid_search.best_estimator_)
        output_file.write('BEST SCORE: ' + '\n')
        output_file.write(grid_search.best_score_)
        output_file.write('BEST CANDIDATE PARAMETER INDEX IN THE CV RESULTS DataFrame: ' + '\n')
        output_file.write(grid_search.best_index_)
        output_file.write(f"---------CONFUSION MATRIX STORED AT PATH : {confusion_matrix_path}-----------" + '\n')
        output_file.write("---------CLASSIFICATION REPORT-----------" + '\n')
        output_file.write(classification_report(y_test, grid_predictions))
        output_file.write("---------END---------" + '\n')
        output_file.write('---------------------------------------------------------------------' + '\n')

def main():

    df = pd.read_csv('../../data/allParticipant_data/allParticipants_5fps_downsampled_preprocessed_norm.csv')
    
    # Specify the output directory
    now = datetime.datetime.now()
    results_directory = '../../results/' + 'RandomForests_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'
        
    # Create 'results_directory' if it doesn't exist
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # RandomForests with GridSearch
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'oob_score': [True, False]
        }
    
    trainModel(df, param_grid, results_directory)

if __name__ == "__main__":
    main()