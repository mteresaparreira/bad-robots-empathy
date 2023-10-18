"""
Description: This python script performs model training on the features extracted from the OpenFace library on the responseVideo dataset of the SuperBAD project.
Model: LogisticRegression with manual 5-fold cross-validation to create folds having non-overlapping participant data.
Author: Sukruth Gowdru Lingaraju
Date Created: September 23rd, 2023
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
from sklearn.linear_model import LogisticRegression


def create_k_fold_DataSplits(df, seed_value = 42):
    """
    This method perform k-fold cross validation for the given participant dataset where-in the participant data is not overlapped in the splits

    Args:
        df (_type_): participant dataframe
        seed_value (int, optional): Defaults to 42.

    Returns:
        numpy.arrays: k-fold data splits 
    """
    
    try:       
        # # Set seed
        random.seed(seed_value)
        np.random.seed(seed_value)
        
        """
        Begin: K-Fold Cross-Validation splits
        
        Identify the range of the splits & assign participants belonging to those ranges to their respective folds
        - 'start_test_indx', 'end_test_indx': defines the range of the particiapants belonging to the 'test_fold'
        - 'test_fold': consists of the participants belonging to the 'k'th fold
        - 'remaining_participants': {set} difference between original 'participants' & 'test_fold' participants
        - 'val_fold': consists of 'val_fold_size' participants randomly shuffled after obtaining 'remaining_participants' belonging to the 'k'th fold
        - 'train_fold': consists of all the remaining participants belonging to the 'k'th fold
        - 'test_folds', 'val_folds', 'train_folds': consists of the set of participants in each fold
        """

        # Identify the unique participants that exist in the dataset
        participants = np.unique(df['participant_id'])

        # Define the number of participants for train, validation, and test
        train_fold_size = 20
        val_fold_size = 3
        test_fold_size = 6

        #number of dataset folds
        num_folds = 5

        # Shuffle the list of participants
        np.random.shuffle(participants)

        # Initialize lists to store train, validation, and test participants for each fold
        train_folds = []
        val_folds = []
        test_folds = []

        # Create non-overlapping test folds and validation folds
        for i in range(num_folds):
            start_test_idx = i * test_fold_size
            end_test_idx = start_test_idx + np.min([test_fold_size, len(participants) - start_test_idx])

            test_fold = participants[start_test_idx:end_test_idx]

            # Identify all the participants except the participants belonging to the test_fold & shuffle them
            remaining_participants = np.setdiff1d(participants, test_fold)
            np.random.shuffle(remaining_participants)

            # Validation set selected from the remaining participants
            val_fold = remaining_participants[:val_fold_size]

            # Identify all the participants that don't belong to 'val_fold' & 'test_fold' and assign them to the 'train_fold'
            train_fold = np.setdiff1d(remaining_participants, val_fold)

            # Append the participant sets to their corresponding folds
            train_folds.append(train_fold)
            val_folds.append(val_fold)
            test_folds.append(test_fold)
        """
        End: K-Fold Cross-Validation splits
        """
    except Exception as e:
        print(f'Exception {e} thrown for: -')
        traceback.print_exc()
        pass

    return train_folds, val_folds, test_folds

def trainModel(superBAD_df, results_directory='.', seedValue = 42):

    df = superBAD_df

    # # Extract features and labels

    # # for naive & naive_n datasets
    # features = df.iloc[:, 3:]
    # target_class = df['class'].values

    # # for full & full_n datasets
    features = df.iloc[:, 4:]
    target_class = df.iloc[:, 2]#.values

    train_folds, val_folds, test_folds = create_k_fold_DataSplits(df, seed_value = 42)

    # print(f'Length of train_folds = {len(train_folds)}')
    # print(f'Length of val_folds = {len(val_folds)}')
    # print(f'Length of test_folds = {len(test_folds)}')

    # print(f'\n Test Fold participants : \n{test_folds}')
    # print(f'\n Train Fold participants : \n{train_folds}')
    # print(f'\n Validation Fold participants : \n{val_folds}')

    # # Train the classifier on the train and validation folds - to determine the best

    with open(f'{results_directory}superBAD_k_fold_cv_logisticRegression.txt', 'a') as output_file:
        output_file.write(f"LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,\
                intercept_scaling=1, l1_ratio=None, max_iter=100,\
                multi_class='multinomial', n_jobs=None, penalty='l1',\
                random_state=42, solver='saga', tol=0.0001, verbose=0,\
                warm_start=False)")
    

    for fold_indx in tqdm(range(len(train_folds)), desc='FOLD', total=len(train_folds)):

        # # Since, we have performed hyper-parameter tuning, we want to now perform cross-validation on those parameters for non-overlapping participants
        # # As a result, trainining participants for a given fold will be = participants in train_folds[fold_index] + validation_fold[fold_index], and the test_participants will be = test_fold[fold_index]
        train_participants = [] 

        # Perform train_participants = train_fold[fold_indx] + val_fold[fold_indx]
        train_participants.extend(train_folds[fold_indx])
        train_participants.extend(val_folds[fold_indx])

        test_participants = test_folds[fold_indx]
        # print(f'Total number of participants in training set = {len(train_participants)}, and in test set = {len(test_participants)}')

        # Split the data into train, and test sets
        train_set_df = df[df['participant_id'].isin(train_participants)]
        X_train = features.loc[train_set_df.index, : ]
        y_train = target_class[train_set_df.index].values
        y_train = y_train.astype('int')

        test_set_df = df[df['participant_id'].isin(test_participants)]
        X_test  = features.loc[test_set_df.index, : ]
        y_test = target_class[test_set_df.index].values
        y_test = y_test.astype('int')

        # print(f'Shape of X_train = {X_train.shape}, Shape of y_train = {y_train.shape}')
        # print(f'Shape of X_test = {X_test.shape}, Shape of y_test = {y_test.shape}')

        classifier = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,
                intercept_scaling=1, l1_ratio=None, max_iter=300,
                multi_class='multinomial', n_jobs=None, penalty='l1',
                random_state=42, solver='saga', tol=0.0001, verbose=10,
                warm_start=False)
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
        confusion_matrix_path = results_directory + f'superbad_logisticRegression_with_k_fold_cv_at_fold_{fold_indx + 1}.png'
        plt.savefig(confusion_matrix_path)
        plt.clf()


        with open(f'{results_directory}superBAD_k_fold_cv_logisticRegression.txt', 'a') as output_file:
            output_file.write('---------------------------------------------------------------------' + '\n')
            output_file.write('Begin Model: LogisticRegression with K-FOLD Cross Validation for Non-overlapping participant data' + '\n')
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
    results_directory = '../../../results/' + 'k_fold_cv_logisticRegression' + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'

    # Create 'results_directory' if it doesn't exist
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    """
    End: Directories specification
    """

    # Invoke the model training method
    trainModel(superBAD_df, results_directory, seedValue = 42)

if __name__ == "__main__":
    main()