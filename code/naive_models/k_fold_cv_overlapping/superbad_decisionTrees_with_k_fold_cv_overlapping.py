"""
Description: This python script performs model training on the features extracted from the OpenFace library on the responseVideo dataset of the SuperBAD project.
Model: DecisionTrees with 5-fold cross-validation to create folds having overlapping participant data.
Author: Sukruth Gowdru Lingaraju
Date Created: September 26th, 2023
Python Version: 3.10.9
Email: sg2257@cornell.edu
"""

import os
import random
import datetime
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

def create_k_fold_DataSplits(df, num_folds=5, seed_value = 42):

    # # Extract features and labels

    # # for naive & naive_n datasets
    # features = df.iloc[:, 3:]
    # target_class = df['class'].values

    # # for full & full_n datasets
    features = df.iloc[:, 4:]
    target_class = df.iloc[:, 2]#.values

    k_fold = KFold(n_splits=num_folds, shuffle=True, random_state=seed_value)

    train_folds = []
    test_folds = []

    # for i, (train_index, test_index) in enumerate(k_fold.split(features)):
    #     print(f"Fold {i}:")
    #     print(f"  Train: index={train_index}")
    #     print(f"  Test:  index={test_index}")

    for train_indexes, test_indexes in k_fold.split(features):
        train_folds.append(train_indexes)
        test_folds.append(test_indexes)

    return train_folds, test_folds

def trainModel(superBAD_df, num_folds=5, results_directory='.', seedValue = 42):

    df = superBAD_df

    # # Extract features and labels

    # # for naive & naive_n datasets
    # features = df.iloc[:, 3:]
    # target_class = df['class'].values

    # # for full & full_n datasets
    features = df.iloc[:, 4:]
    target_class = df.iloc[:, 2]#.values

    train_folds, test_folds = create_k_fold_DataSplits(df, num_folds, seed_value = seedValue)

    # print(f'Length of train_folds = {len(train_folds)}')
    # print(f'Length of test_folds = {len(test_folds)}')

    # print(f'\n Test Fold participants : \n{test_folds}')
    # print(f'\n Train Fold participants : \n{train_folds}')


    # # # Train the classifier on the train and validation folds - to determine the best

    with open(f'{results_directory}superBAD_k_fold_cv_overlapping_data_decisionTrees.txt', 'a') as output_file:
        output_file.write(f"Model Selection: classifier = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',\
                    max_depth=None, max_features='auto', max_leaf_nodes=None,\
                    min_impurity_decrease=0.0, min_impurity_split=None,\
                    min_samples_leaf=1, min_samples_split=2,\
                    min_weight_fraction_leaf=0.0, presort='deprecated',\
                    random_state=42, splitter='best')")
    

    for fold_indx in tqdm(range(len(train_folds)), desc='FOLD', total=len(train_folds)):

        # # Since, we have performed hyper-parameter tuning, we want to now perform cross-validation on those parameters for non-overlapping participants
        # # As a result, trainining participants for a given fold will be = participants in train_folds[fold_index], and the test_participants will be = test_fold[fold_index]
        train_participants = [] 

        # Perform train_participants = train_fold[fold_indx]
        train_participants.extend(train_folds[fold_indx])
        test_participants = test_folds[fold_indx]
        # print(f'Total number of participants in training set = {len(train_participants)}, and in test set = {len(test_participants)}')

        # Split the data into train, and test sets
        X_train = features.loc[train_participants, : ]
        y_train = target_class[train_participants].values
        y_train = y_train.astype('int')

        X_test  = features.loc[test_participants, : ]
        y_test = target_class[test_participants].values
        y_test = y_test.astype('int')

        # print(f'Shape of X_train = {X_train.shape}, Shape of y_train = {y_train.shape}')
        # print(f'Shape of X_test = {X_test.shape}, Shape of y_test = {y_test.shape}')

        classifier = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='entropy',
                    max_depth=None, max_features='auto', max_leaf_nodes=None,
                    min_impurity_decrease=0.0, min_impurity_split=None,
                    min_samples_leaf=1, min_samples_split=2,
                    min_weight_fraction_leaf=0.0, presort='deprecated',
                    random_state=42, splitter='best')
        classifier.fit(X_train, y_train)

        y_predict = classifier.predict(X_test)

        # Calculate the confusion matrix after training models through gridsearch
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
        confusion_matrix_path = results_directory + f'superbad_decisionTrees_with_k_fold_cv_overlapping_data_at_fold_{fold_indx + 1}.png'
        plt.savefig(confusion_matrix_path)
        plt.clf()


        with open(f'{results_directory}superBAD_k_fold_cv_overlapping_data_decisionTrees.txt', 'a') as output_file:
            output_file.write('---------------------------------------------------------------------' + '\n')
            output_file.write('Begin Model: DecisionTrees with K-FOLD Cross Validation for Overlapping Participant Data' + '\n')
            output_file.write(f'--------------------------- FOLD = {fold_indx} ----------------------------' + '\n')
            output_file.write('Dataset: allParticipants_5fps, full_n' + '\n')
            # output_file.write('---------------------------------------------------------------------' + '\n')
            # output_file.write(f'# of FEATURES SEEN DURING MODEL FITTING: {str(len(classifier.feature_names_in_))}' + '\n')
            # output_file.write('---------------------------------------------------------------------' + '\n')
            # output_file.write('NAMES OF FEATURES SEEN DURING MODEL FITTING: ' + '\n')
            # output_file.write(str(classifier.feature_names_in_) + '\n')
            # output_file.write('---------------------------------------------------------------------' + '\n')
            # output_file.write('FEATURES IMPORTANCE SEEN DURING MODEL FITTING: ' + '\n')
            # output_file.write(str(classifier.feature_importances_) + '\n')
            # output_file.write('---------------------------------------------------------------------' + '\n')
            # output_file.write(f'BEST ESTIMATOR: {str(classifier.estimator_)}' + '\n')
            output_file.write('---------------------------------------------------------------------' + '\n')
            output_file.write(f"CONFUSION MATRIX STORED AT PATH : {confusion_matrix_path} " + '\n')
            output_file.write("---------CLASSIFICATION REPORT-----------" + '\n')
            output_file.write(classification_report(y_test, y_predict))
            output_file.write("------------------END------------------" + '\n')
        # break # break after one fold

def main():

    """
    Begin: Directories specification
    """

    # allParticipants dataset path
    participant_data_path = '../../../data/allParticipant_data/allParticipants_5fps_downsampled_preprocessed_norm.csv'
    superBAD_df = pd.read_csv(participant_data_path)

    # results directory - make a new folder with the day and time of the run
    now = datetime.datetime.now()
    results_directory = '../../../results/' + 'k_fold_cv_overlapping_data_decisionTrees_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'

    # Create 'results_directory' if it doesn't exist
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    """
    End: Directories specification
    """

    # Invoke the model training method
    num_folds = 5
    trainModel(superBAD_df, num_folds, results_directory, seedValue = 42)

if __name__ == "__main__":
    main()