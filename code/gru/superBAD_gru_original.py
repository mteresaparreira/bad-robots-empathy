# superBAD_gru.py

"""
Description: This script trains a GRU Model on the facial data extracted from the OpenFace library to classify human reactions into Control, Failure Human, Failure Robot classes
Author: Sukruth Gowdru Lingaraju
Date Created: August 18th, 2023
Python Version: 3.10.9
Email: sg2257@cornell.edu
"""

"""
Begin: Import Dependencies 
"""

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
from tensorflow.keras.layers import GRU, Dense, Dropout, Bidirectional

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

def createDataSplits(df, results_directory= '.', seed_value = 42, sequence_length = 1):

    """
    createDataSplits(): accepts a dataframe along with the directory to store the results and sequence_length - to perform data splits for training, validation, and testing

    Parameters:
    - df
    - results_directory: to write the 'Exception Error' thrown if there are any problems in splitting the data
    - seed_value
    - sequence_length (a.k.a: lookbacks)
    """
    
    try:
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
        target_class = df.iloc[:, 2].values
        target_class = target_class.astype('int')
        
        """
        Begin: K-Fold Cross-Validation splits
        
        Identify the range of the splits & assign participants belonging to those ranges to their respective folds
        - 'start_test_indx', 'end_test_indx': defines the range of the particiapants belonging to the 'test_fold'
        - 'test_fold': consists of the participants belonging to the 'k'th fold
        - 'remaining_participants': set difference between original 'participants' & 'test_fold' participants
        - 'val_fold': consists of 'val_fold_size' participants randomly shuffled after obtaining 'remaining_participants' belonging to the 'k'th fold
        - 'train_fold': consists of all the remaining participants belonging to the 'k'th fold
        - 'test_folds', 'val_folds', 'train_folds': consists of the set of participants in each fold
        """
        
        participants = np.unique(df['participant_id'])

        #Number of participants for train, validation, and test
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

        # # Create train, validation, and test sets for each fold in 'num_folds'
        # For now, do only for fold: '0'
        train_fold = train_folds[0]
        val_fold = val_folds[0]
        test_fold = test_folds[0]
        
        """
        Begin: train, val, test: splits & sequences
        """

        # Split the data into train, validation, and test sets
        train_set = df[df['participant_id'].isin(train_fold)]
        X_train = features.loc[train_set.index, : ]
        
        val_set = df[df['participant_id'].isin(val_fold)]
        X_val = features.loc[val_set.index, : ]
        
        test_set = df[df['participant_id'].isin(test_fold)]
        X_test  = features.loc[test_set.index, : ]

        # Convert labels to categorical format
        num_classes = np.max(target_class) + 1  # Assuming labels start from 0
        labels_ohe = np.eye(num_classes)[target_class]
        
        # Retrieve y_train, y_val, and y_test: values corresponding to same indexes, from labels_ohe
        y_train = labels_ohe[X_train.index]
        y_val = labels_ohe[X_val.index]
        y_test = labels_ohe[X_test.index]

#         # Print size of all sets
#         print('Size of all sets before resetting the X indexes')
#         print(X_train.shape, y_train.shape)
#         print(X_val.shape, y_val.shape)
#         print(X_test.shape, y_test.shape)

        #reset indexes
        X_train = X_train.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        
#         # Print size of all sets after resetting the X indexes
#         print('Size of all sets after resetting the X indexes')
#         print(X_train.shape, y_train.shape)
#         print(X_val.shape, y_val.shape)
#         print(X_test.shape, y_test.shape)
        
        #### Split data into train and test sets: if k-fold cross-validation is not needed
        ### X_train, X_test, y_train, y_test = train_test_split(features, labels_ohe, test_size=0.2, random_state=seed_value)
        
        """
        Begin: Sequence Creation as per defined 'sequence_length'(a.k.a: lookbacks)
        """
        
        X_train_sequences = [X_train[i : i + sequence_length] for i in range(len(X_train) - sequence_length + 1)]
        y_train_sequences = y_train[sequence_length - 1 : ]

        X_val_sequences = [X_val[i : i + sequence_length] for i in range(len(X_val) - sequence_length + 1)]
        y_val_sequences = y_val[sequence_length - 1 : ]

        X_test_sequences = [X_test[i : i + sequence_length] for i in range(len(X_test) - sequence_length + 1)]
        y_test_sequences = y_test[sequence_length - 1 : ]
        
        """
        End: Sequence Creation
        """
        """
        End: train, val, test: splits & sequences
        """
        return num_classes, X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences

    except Exception as e:
        with open(f'{results_directory}/results_GRU.txt', 'a') as results_file:
            results_file.write(
                f'Exception {e} thrown during splitting the dataset for :- '
                f'{traceback.print_exc()}'
            )
        pass

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

def executeModel_GRU(df, results_directory, seed_value = 42, sequence_lengths = 1, units = 64, dropouts = 0.2, activations = 'softmax', losses = 'categorical_crossentropy', optimizers = 'adam', epochs = 100, batch_sizes = 32):

    """
    executeModel_GRU(): takes in the dataFrame along with the directory specification & hyper-parameters and trains a GRU model

    Parameters:
    - df
    - results_directory
    - seed_value
    - sequence_length (a.k.a: lookback)
    - units
    - dropouts
    - activations
    - losses
    - optimizers
    - epochs
    """
    # Keep count of the number of different search combinations
    search_count = 0

    # Create a figure with subplots for plotting the training and validation accuracy for various batch_sizes against the # of epochs
    num_rows = 2
    num_cols = 2
    batch_figure, axs = plt.subplots(num_rows, num_cols, figsize=(15, 8))
    
    tf.compat.v1.reset_default_graph()

    for sequence_length in sequence_lengths:
        for unit in units:
            for dropout in dropouts:
                for activation in activations:
                    for loss in losses:
                        for optimizer in optimizers:
                            for epoch in epochs:
                                for i, batch_size in enumerate(batch_sizes):
                                    try:
                                        search_count += 1
                                        
                                        # # Retrieve the splits
                                        num_classes, X_train, X_val, X_test, y_train, y_val, y_test, X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences, X_test_sequences, y_test_sequences = createDataSplits(df, results_directory, seed_value, sequence_length)
                                        
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

                                            # # Create the GRU model
                                            model = Sequential()
                                            model.add(GRU(units = unit, input_shape = (sequence_length, X_train.shape[1])))
                                            model.add(Dropout(dropout))
                                            model.add(Dense(units=num_classes, activation = activation))

                                            # Compile the model
                                            model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

                                            # Train the model and capture the history data
                                            model_history = model.fit(
                                                np.array(X_train_sequences),
                                                y_train_sequences,
                                                batch_size = batch_size,
                                                epochs = epoch,
                                                verbose = '2',
                                                validation_data=(np.array(X_val_sequences), y_val_sequences),
                                            )
                                            
                                            # Obtain the training loss & accuracy data
                                            train_loss, train_accuracy = model_history.history['loss'], model_history.history['accuracy']
                                            
                                            # Obtain the validation loss & accuracy data
                                            val_loss, val_accuracy = model_history.history['val_loss'], model_history.history['val_accuracy']
                                            
                                            # Evaluate the model on test data
                                            test_loss, test_accuracy = model.evaluate(np.array(X_test_sequences), y_test_sequences)

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
                                                'val_loss': val_loss,
                                                'val_accuracy': val_accuracy,
                                                'test_loss': test_loss,
                                                'test_accuracy': test_accuracy
                                            }

                                            with open(model_data_path + f'model_{sequence_length}_{unit}_{dropout}_{activation}_{loss}_{optimizer}_{epoch}_{batch_size}', 'wb') as f:
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

                                            y_predict_probs = model.predict(np.array(X_test_sequences))
                                            y_predict = np.argmax(y_predict_probs, axis=1)  # Convert to class labels

                                            report = classification_report(np.argmax(y_test_sequences, axis=1), y_predict)

                                            """
                                                Generate the Confusion Matrix
                                            """

                                            # Calculate the confusion matrix
                                            conf_matrix = confusion_matrix(np.argmax(y_test_sequences, axis=1), y_predict)

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
                                            
                                            confusion_matrix_path = results_directory + f'confusion_matrices/confusion_matrix_{sequence_length}_{unit}_{dropout}_{activation}_{loss}_{optimizer}_{epoch}_{batch_size}.png'
                                            plt.savefig(confusion_matrix_path)
                                            plt.clf()
                                            
                                            # # Plot training and validation accuracy plots for varying batch_sizes
                                            # Plot training and validation accuracy on the current subplot

                                            # Calculate row and column indices for the current subplot
                                            row_idx = i // num_cols
                                            col_idx = i % num_cols
                                            
                                            # Get the current subplot
                                            ax = axs[row_idx, col_idx]
                                            ax.clear() # clear the legends from previous plots
                                            plot_batch_size_accuracy(train_accuracy, val_accuracy, batch_size, ax)

                                            """
                                            ------------------------------------------------------------------------------------
                                            Begin: Logging 
                                            - Write all the information of the particular combination of the model to a file below
                                            """

                                            with open(f'{results_directory}/results_GRU.txt', 'a') as results_file:
                                                results_file.write("\n")
                                                results_file.write(f"------------ NETWORK: GRU ------------" + "\n")
                                                results_file.write(f"------------ BEGIN SEARCH : {search_count} ------------" + "\n")
                                                results_file.write("------------ TYPE ------------" + "\n")

                                                results_file.write(
                                                    f'Sequence Length = {sequence_length}\n'
                                                    f'Units = {unit}\n'
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
                                                results_file.write(f'Validation Loss: {val_loss[-1]: .4f}' + '\n')
                                                results_file.write(f'Validation Accuracy: {val_accuracy[-1]: .4f}' + '\n')
                                                results_file.write(f'Test Loss: {test_loss:.4f}' + '\n')
                                                results_file.write(f'Test Accuracy: {test_accuracy:.4f}' + '\n')

                                                results_file.write("------------ PARAMETERS ------------" + "\n")

                                                results_file.write(f'Model Parameters: {model_history.params}' + '\n')
                                                results_file.write(f'Model Keys: {model_history.history.keys()}' + '\n')

                                                results_file.write("------------ CLASSIFICATION REPORT ------------" + "\n")

                                                results_file.write(report + '\n')

                                                results_file.write("------------ CONFUSION MATRIX ------------" + "\n")

                                                results_file.write(f'Confusion matrix saved for confusion_matrix_{sequence_length}_{unit}_{dropout}_{activation}_{loss}_{optimizer}_{epoch}_{batch_size}.png' + '\n')
                                                results_file.write(f"------------ END SEARCH : {search_count} ------------" + "\n")
                                                results_file.write("\n")

                                    except Exception as e:
                                        with open(f'{results_directory}/results_GRU.txt', 'a') as results_file:
                                            results_file.write(
                                                f'Exception {e} thrown for :- \n'
                                                f'{traceback.print_exc()} \n'
                                                f'Sequence Length = {sequence_length}\n'
                                                f'Units = {unit}\n'
                                                f'Dropout = {dropout}\n'
                                                f'Activation = {activation}\n'
                                                f'Loss Function = {loss}\n'
                                                f'Optimizer = {optimizer}\n'
                                                f'Epochs = {epoch}\n'
                                                f'Batch Size = {batch_size}\n'
                                                f'Seed Value = {seed_value}\n'
                                            )
                                    # break # batch_size break
                                """
                                BEGIN: Subplots for training and validation accuracy for varying batch_sizes
                                """
                                # Adjust layout and save the figure for training & validation accuracy for varying batch_sizes
                                batch_size_results_path = results_directory + 'batch_size_results/'
                    
                                if not os.path.exists(batch_size_results_path):
                                    os.makedirs(batch_size_results_path)

                                batch_size_accuracy_path = batch_size_results_path + f'batch_size_comparison_{sequence_length}_{unit}_{dropout}_{activation}_{loss}_{optimizer}_{epoch}.png'
                                batch_figure.tight_layout(pad=2.5)
                                batch_figure.savefig(batch_size_accuracy_path)
                                # plt.show()
                                plt.close(batch_figure)
                                plt.clf()

                                with open(f'{results_directory}/results_GRU.txt', 'a') as results_file:
                                    results_file.write("------------ BATCH_SIZE PLOTS ------------" + "\n")
                                    results_file.write(f'BATCH_SIZE plots saved for batch_size_comparison_{sequence_length}_{unit}_{dropout}_{activation}_{loss}_{optimizer}_{epoch}.png' + '\n')
                                    results_file.write("------------ BATCH_SIZE PLOTS ------------" + "\n")
                                """
                                End: Logging 
                                ------------------------------------------------------------------------------------
                                End: batch_size subplots
                                """
        #                         break # epochs break
        #                     break # optimizers break
        #                 break # losses break
        #             break # activation functions break
        #         break # dropouts break
        #     break # units break
        # break # sequence_lengths break

def main():
    
    """
    Begin: Directories specification
    """
    
    # allParticipants dataset path
    superBAD_df = pd.read_csv('../../data/allParticipant_data/allParticipants_5fps_downsampled_preprocessed_norm.csv')
    
    # results directory - make a new folder with the day and time of the run
    import datetime
    now = datetime.datetime.now()
    results_directory = '../../results/' + 'GRU_' + now.strftime("%Y-%m-%d_%H-%M-%S") + '/'
    
    # Create 'results_directory' if it doesn't exist
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    
    """
    End: Directories specification
    """

    """
    Begin: Hyperparameters definition
    """
    
    sequence_lengths = [5, 10, 15]
    units = [32, 64, 128]
    dropouts = [0, 0.2, 0.4, 0.6]
    activations = ['sigmoid', 'relu', 'tanh', 'softmax']
    losses = ['categorical_crossentropy', 'binary_crossentropy', 'hinge']
    optimizers = ['SGD', 'Adam']
    epochs = [250, 500, 750, 1000]
    batch_sizes = [512, 1024, 2048, 4096]

    """
    End: Hyperparameters definition
    """
    
    """
    Begin: Call methods
    """

    # Call your model execution function with keyword arguments
    executeModel_GRU(
        superBAD_df,
        results_directory,
        seed_value = 42,
        sequence_lengths = sequence_lengths,
        units = units,
        dropouts = dropouts,
        activations = activations,
        losses = losses,
        optimizers = optimizers,
        epochs = epochs,
        batch_sizes = batch_sizes
    )
    
    """
    End: Call methods
    """
"""
End: Method Definitions
"""

if __name__ == "__main__":
    main()