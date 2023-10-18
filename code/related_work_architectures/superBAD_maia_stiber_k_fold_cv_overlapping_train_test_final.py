# superBAD_maia_stiber_k_fold_cv_overlapping_train_val_plot.py

"""
Description: This script trains a related Model on the facial data extracted from the OpenFace library to classify human reactions into Control, Failure Human, Failure Robot classes as per the architecture mentioned in the paper - 'On Using Social Signals to Enable Flexible Error-Aware HRI'
Author: Sukruth Gowdru Lingaraju
Date Created: September 26th, 2023
Python Version: 3.10.9
Email: sg2257@cornell.edu
"""

"""
Begin: Import Dependencies 
"""
import gc
import os
import random
import datetime
import pickle
import traceback
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import KFold


"""
End: Import Dependencies 
"""
"""
Begin: Method Definitions
"""
# def classification_report_tolerance(y_pred, y_true, margin = 1):
    
#     metrics_dict = dict()
#     all_metrics_dict = dict()
#     classes = np.unique(y_true)
#     all_p = []
#     all_r = []
#     all_a = []
#     all_f1 = []
    
#     for classi in classes:
#         tp = 0
#         tn = 0
#         fp = 0
#         fn = 0


#         for i, y in enumerate(y_pred):
#             if y == classi:
#                 if y in y_true[i-margin:i+margin+1]:
#                     #print(y_true[i-1:i+1].shape)
#                     tp = tp + 1
#                 else:
#                     fp = fp + 1
#             else:
#                 if y not in y_true[i-1:i+1]:
#                     fn = fn + 1
#                 else:
#                     tn = tn + 1
#         precision = tp/(tp+fp)
#         recall = tp / (tp + fn)
#         f1 = (2*precision*recall)/(precision+recall)
#         accuracy = (tp+tn)/(tp+tn+fp+fn)
#         metrics_dict[classi] = [tp, tn, fp, fp]
#         all_metrics_dict[classi] = [precision, recall, f1, accuracy]
#         all_p.append(precision)
#         all_r.append(recall)
#         all_a.append(accuracy)
#         all_f1.append(f1)
        
    
#     print(metrics_dict)
#     print(all_metrics_dict)
    
#     macro_dict = dict()
#     macro_dict['macro-precision'] = sum(np.array(all_p))/len(all_p)
#     macro_dict['macro-recall'] = sum(np.array(all_r))/len(all_r)
#     macro_dict['macro-accuracy'] = sum(np.array(all_a))/len(all_a)
#     macro_dict['macro-f1'] = (2* macro_dict['macro-precision']*macro_dict['macro-recall'])/(macro_dict['macro-recall'] + macro_dict['macro-precision'])
    
#     print(macro_dict)
    
#     return metrics_dict, all_metrics_dict, macro_dict

def plot_batch_size_accuracy(train_accuracy, val_accuracy, batch_size, axs):
    """
    plot_batch_size_accuracy(): creates subplots of training & validation accuracy scores for varying batch sizes
    """
    epochs = range(1, len(train_accuracy) + 1)
    axs.plot(epochs, train_accuracy, label='Training Accuracy')
    axs.plot(epochs, val_accuracy, label='Validation Accuracy')
    axs.set_title(f'Batch Size: {batch_size}')
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Accuracy')
    axs.legend()

def create_k_fold_DataSplits(df, num_folds=5, seed_value = 42):

    # # Set seed
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
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

    return features, target_class, train_folds, test_folds

def executeModel(df, nums_folds = 5, results_directory='./', seed_value = 42, dropout = 0.2, activation = 'softmax', loss = 'categorical_crossentropy', optimizer = 'adam', epoch = 100, batch_size = 32):

    """
    executeModel(): takes in the dataFrame along with the directory specification & hyper-parameters and trains a model on the best performing hyper-parameter on a k-fold cross validation with overlapping participant data

    Parameters:
    - df
    - results_directory
    - seed_value
    - dropouts
    - activations
    - losses
    - optimizers
    - epochs
    """
    
    features, target_class, train_folds, test_folds = create_k_fold_DataSplits(df, nums_folds, seed_value)
    
    num_rows = 3
    num_cols = 2
    batch_figure, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))

    with open(f'{results_directory}/k_fold_cv_overlapping_data_results_related_work_maia_stiber_arch.txt', 'a') as results_file:
        results_file.write("Related work arch: Maia Stiber - Mixed participants data + 5 fold cross validation on the best performing model" + "\n")

    for fold_indx in range(nums_folds):
        try:
            """
            Begin: train, test: splits 
            """

            # # Since, we have performed hyper-parameter tuning, we want to now perform cross-validation on those parameters for non-overlapping participants
            # # As a result, trainining participants for a given fold will be = participants in train_folds[fold_index], and the test_participants will be = test_fold[fold_index]
            train_participants = [] 

            # Perform train_participants = train_fold[fold_indx]
            train_participants = train_folds[fold_indx]
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

            # Convert labels to categorical format
            num_classes = np.max(target_class) + 1  # Assuming labels start from 0
            labels_ohe = np.eye(num_classes)[target_class]
            
            # Retrieve y_train, y_val, and y_test: values corresponding to same indexes, from labels_ohe
            y_train = labels_ohe[X_train.index]
            y_test = labels_ohe[X_test.index]

            # # Print size of all sets
            # print('Size of all sets before resetting the X indexes')
            # print(X_train.shape, y_train.shape)
            # print(X_test.shape, y_test.shape)
            # print(f'X_train before resetting the X indexex: \n {X_train}')


            #reset indexes
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            
            # # Print size of all sets after resetting the X indexes
            # print('Size of all sets after resetting the X indexes')
            # print(X_train.shape, y_train.shape)
            # print(X_test.shape, y_test.shape)
            # print(f'X_train after resetting the X indexex: \n {X_train}')

            """
            End: train, test: splits
            """
            """
            ------------------------------------------------------------------------------------
            Begin: Model Architecture
            """
            # # Check GPU availability
            # gpus = tf.config.list_physical_devices('GPU')
            # print("GPU available:", gpus)
            # # tf.debugging.set_log_device_placement(True)

            # # Assuming there is at least one GPU available
            # if gpus:
            #     # Set TensorFlow to use GPU 0
            #     try:
            #         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            #         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #         # gpu_device = tf.config.list_physical_devices('GPU')[0]
            #         # tf.config.experimental.set_memory_growth(gpu_device, True)
            #         # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            #     except RuntimeError as e:
            #         print(e)
                    
            # # Specify the device used for computation
            with tf.device('/GPU:0'):  # Use GPU 0

                # Build and train your model here

                # # Create the related_work_maia_stiber_arch model
                model = Sequential()
                # Add the hidden layers
                model.add(Dense(units=64, input_shape=(49, ), activation=activation))
                model.add(Dense(units=128, activation=activation))
                model.add(Dense(units=64, activation=activation))

                # Add dropout layer to prevent overfitting
                model.add(Dropout(dropout))

                # Add the output layer
                model.add(Dense(units=num_classes, activation = activation))

                # Compile the model
                model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

                # Train the model and capture the history data
                model_history = model.fit(
                    np.array(X_train),
                    y_train,
                    batch_size = batch_size,
                    epochs = epoch,
                    verbose = '2'
                )

                # Obtain the training loss & accuracy data
                train_loss, train_accuracy = model_history.history['loss'], model_history.history['accuracy']
                
                # Evaluate the model on test data
                test_loss, test_accuracy = model.evaluate(np.array(X_test), y_test)

                """
                Save the model information data as an object
                Define the path to store the object data & create the directory if it does not exist
                """
                model_data_path = results_directory + 'model_data/'
                
                if not os.path.exists(model_data_path):
                    os.makedirs(model_data_path)

                # Store the model learning data using pickle
                model_data_information = {
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy
                }

                with open(model_data_path + f'model_{fold_indx}_{dropout}_{activation}_{loss}_{optimizer}_{epoch}_{batch_size}', 'wb') as f:
                    pickle.dump(model_data_information, f)
                
                """
                End: Model Architecture
                ------------------------------------------------------------------------------------
                """
                """
                ------------------------------------------------------------------------------------
                    Predictions using the Model
                    ===========================

                    When making predictions using the trained model, the output is in the form of predicted probabilities,
                    indicating the likelihood of each sample belonging to each target class.

                    Predicted Probabilities (y_predict_probs):
                    - Shape: (#samples, #target_classes)
                    - Each value in y_predict_probs represents the probability of the corresponding sample being classified
                    into the respective class.

                    Converting Probabilities to Class Labels (y_predict):
                    - The y_predict array is derived by finding the index of the maximum value along a specified axis.
                    - It represents the predicted class label for each sample based on the highest predicted probability.
                ------------------------------------------------------------------------------------
                """

                y_predict_probs = model.predict(np.array(X_test))
                y_predict = np.argmax(y_predict_probs, axis=1)  # Convert to class labels

                report = classification_report(np.argmax(y_test, axis=1), y_predict)

                """
                    Generate the Confusion Matrix
                """

                # Calculate the confusion matrix
                conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_predict)

                # Get the class labels (assuming y_true and y_pred are integer class labels)
                class_labels = ['Control', 'Failure_Human', 'Failure_Robot']

                # Plot the confusion matrix using seaborn and matplotlib
                plt.figure(figsize=(8, 8))
                sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix")

                # Save the confusion matrix as an image
                confusion_matrix_path = results_directory + f'confusion_matrices/'
                
                if not os.path.exists(confusion_matrix_path):
                    os.makedirs(confusion_matrix_path)
                
                confusion_matrix_path = results_directory + f'confusion_matrices/confusion_matrix_{fold_indx}_{dropout}_{activation}_{loss}_{optimizer}_{epoch}_{batch_size}.png'
                plt.savefig(confusion_matrix_path)
                plt.clf()

                """
                ------------------------------------------------------------------------------------
                Begin: Logging 
                - Write all the information of the particular combination of the model to a file below
                """

                with open(f'{results_directory}/k_fold_cv_overlapping_data_results_related_work_maia_stiber_arch.txt', 'a') as results_file:
                    results_file.write("\n")
                    results_file.write(f"------------ BEGIN CV FOLD : {fold_indx} ------------" + "\n")
                    results_file.write("------------ TYPE ------------" + "\n")
                    results_file.write(
                        f'Dropout = {dropout}\n'
                        f'Activation = {activation}\n'
                        f'Loss Function = {loss}\n'
                        f'Optimizer = {optimizer}\n'
                        f'Epochs = {epoch}\n'
                        f'Batch Size = {batch_size}\n'
                        f'Seed Value = {seed_value}\n'
                    )
                    results_file.write("------------ METRICS ------------" + "\n")
                    results_file.write(f'Training Loss: {train_loss[-1]: .4f}' + '\n')
                    results_file.write(f'Training Accuracy: {train_accuracy[-1]: .4f}' + '\n')
                    results_file.write(f'Test Loss: {test_loss:.4f}' + '\n')
                    results_file.write(f'Test Accuracy: {test_accuracy:.4f}' + '\n')
                    results_file.write("------------ PARAMETERS ------------" + "\n")
                    results_file.write(f'Model Parameters: {model_history.params}' + '\n')
                    results_file.write(f'Model Keys: {model_history.history.keys()}' + '\n')
                    results_file.write("------------ CLASSIFICATION REPORT ------------" + "\n")
                    results_file.write(report + '\n')
                    results_file.write("------------ CONFUSION MATRIX ------------" + "\n")
                    results_file.write(f'Confusion matrix saved for confusion_matrix_{dropout}_{activation}_{loss}_{optimizer}_{epoch}_{batch_size}.png' + '\n')
                    results_file.write("------------ END ------------" + "\n")
                    results_file.write("\n")
                    
                    # clear up memory
                    del model_history

        except Exception as e:
            with open(f'{results_directory}/k_fold_cv_overlapping_data_results_related_work_maia_stiber_arch.txt', 'a') as results_file:
                results_file.write(
                    f'Exception {e} thrown for :- \n'
                    f'{traceback.print_exc()} \n'
                    f'Dropout = {dropout}\n'
                    f'Activation = {activation}\n'
                    f'Loss Function = {loss}\n'
                    f'Optimizer = {optimizer}\n'
                    f'Epochs = {epoch}\n'
                    f'Batch Size = {batch_size}\n'
                    f'Seed Value = {seed_value}\n'
                )
        # clear up memory
        tf.keras.backend.clear_session()
        gc.collect()
    """
    End: Logging 
    ------------------------------------------------------------------------------------
    """

def main():
    """
    Begin: Directories specification
    """
    
    # allParticipants dataset path
    superBAD_df = pd.read_csv('../../data/allParticipant_data/allParticipants_5fps_downsampled_preprocessed_norm.csv')
    
    # results directory - make a new folder with the day and time of the run
    import datetime
    now = datetime.datetime.now()
    results_directory = '../../results/' + 'k_fold_cv_overlapping_data_related_work_maia_stiber_arch_consider_only_plots_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'
    
    # Create 'results_directory' if it doesn't exist
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    """
    End: Directories specification
    """

    """
    Begin: Hyperparameters definition
    """
    nums_folds = 5
    dropouts = 0 # [0, 0.2, 0.4, 0.6]
    activations = 'sigmoid' #['relu', 'tanh', 'softmax']
    losses = 'categorical_crossentropy' #['categorical_crossentropy'# , 'sparse_categorical_crossentropy', 'binary_crossentropy', 'hinge']
    optimizers = 'SGD' #['SGD', 'Adam']
    epochs = 10000 #[250, 500, 1000]
    batch_sizes = 4096 # [512, 1024, 2048, 4096]

    """
    End: Hyperparameters definition
    """
    
    """
    Begin: Call methods
    """

    # Call your model execution function with keyword arguments
    executeModel(
        superBAD_df,
        nums_folds,
        results_directory,
        seed_value = 42,
        dropout = dropouts,
        activation = activations,
        loss = losses,
        optimizer = optimizers,
        epoch = epochs,
        batch_size = batch_sizes
    )

    """
    End: Call methods
    """

if __name__ == "__main__":
    main()