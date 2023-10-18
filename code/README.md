### This Directory consists of the following:


* Preprocessing: Scripts that were used in the preprocessing of the :
    * Stimulus Videos 
    * Participant's Survey Logs
    * Participant's Facial Features
    * Creating the final Datasets
    * Obtaining Metadata etc.
    * Feature downsampling (based on a given frequency)

* Analysis: consists of scripts that are used to check for the top `n` models after performing hyper-parameter tuning on the models.

*  Naive Machine Learning Models: where the models are used from the [sklearn](https://scikit-learn.org/stable/) library.  
    *  NaiveBayes
    * Logistic Regression
    * Support Vector Machines
    * Decision Trees
    * Random Forests

* A more complex Deep Learning Networks are scripted from the [keras](https://keras.io/api/layers/) libaray:  

    * LSTMs
    * BiLSTMs
    * GRUs
    * Transformer 

    Each model has the following scripts:

    - `superBAD_model_original.py`: that has the code to execute the model on the participant data on a number of hyper-parameter combinations to determine the best performing parameter combination.
    - `superBAD_model_k_fold_cv_overlapping_train_val_plot.py`: has the code to execute the model on a given parameter combination on a 5 FOLD cross validation - where the folds have overlapping participant data. This code is used to obtain the plots during the 5 fold CV - to see how the model behaves.  
    Here, the validation dataset is used as the test_data during model training.

    - `superBAD_model_k_fold_cv_overlapping_train_test_final.py`: has the code to execute the model on the final parameter combination (usually executed after observing the plots obtained from `superBAD_model_k_fold_cv_overlapping_train_val_plot.py` to see how the model behaves) on a 5 FOLD cross validation - where the folds have overlapping participant data.  
    Here, the test dataset is used as the test_data during model training - without any validation data provided for the model.

    - `extract_k_fold_cv_model_output_information.py`: has the code to read through the output log and extract the information into a dataframe - to determine the model's performance and inference.  

    For faster model execution, the models were parallezied to run based on the `sequence_length` (i.e.: `lookback` length) - as a result, there are files in the `analysis` - consists of scripts - `model_script_#_initial_analysis.ipynb`.