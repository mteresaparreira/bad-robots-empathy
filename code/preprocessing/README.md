### The preprocessing directory consists of code that perform the following:



* `initial_preprocessing/`  
    * `feature_stats.ipynb`: calculates statistical values on the features extracted from the participant responses of SuperBAD
    * `match_participantID_prolificID.ipynb`:
        1. Matches the Prolific ID to the Participant ID of the survey users & extracts the duration of time taken by the users to complete the survey
        1. Calculates the Attention Score of the survey users with respect to the stimulus video
        1. Appends the newUsers to the participant_log of the existing users
        1. Extracts the number of responses
        This information has been obtained through the Survey User Response form exported from Qualtrics
    * `stimulusVideo_preprocessing.ipynb`: 
        1. Identifies the final stimulusVideos that are requried to be used in the the study
        1. Performs:
            - Audio Removal
            - Renaming
            - Addition of a lag (i.e.: black screen at the end of the stimulusVideo)
    * `webm to mp4 conversion.ipynb`: 
        1. Identifies all the studyData's survey response recordings
        1. Iterates over all the responses and converts each of its recordings from .webm to .mp4 video file format
        1. Identifies all the responses that are repeated for a particular stimulusVideo

* `extract_model_stat_features.ipynb`: From the features extracted from OpenFace   
    - it drops all the unnecessary features and retains only the `required_features`. 
    - It then performs Class Label Mapping - as per:

    ```
    class_types = {
        'control': 0,
        'failure_robot': 1,
        'failure_human': 2
    }
    ```
    - Normalize the features

* `gridSearch_results_logs_to_dataframe.ipynb`: converts the logs generated whilst performing hyper-parameter tuning for complex model architecture & converts the important information from the logs to a value in a dataframe

* `metadata_openFace_dataset.ipynb`: calculates metadata on the - `allParticipants`, `(downsampled to) 5 FPS`, `normalized` dataset that is used in model training for the SuperBAD study

* `metadata_stimulus_videos.ipynb`: calculates various metadata about the videos collected for the SuperBAD study  
    - Stimulus Videos
    - Response Videos (of the study participants)

* `metadata_study_participant_demographic_info.ipynb`:  calculate various metadata on the information provided by the participants during the survey, such as:
    - Age
    - Gender
    - Education Level
    - Ethnicity
    - Nationality
    - Language Proficiency
    - Survey Duration period

* `openFace_features_downsampling.ipynb`: performs downsampling on the OpenFace Extracted features' dataframes

    * Downsampling - to a specified `sampling_size`
    * Preprocesses the downsampled data
        - Excludes datapoints in `failure videos` which are before the failureOccurrence_timestamp
    * Merges `all participants all videos` that are downsampled to a particular `sampling_size` and preprocessed.

* `superbad_gradualFailure_counts.py`: obtains the total number of Sudden and Gradual Videos that are present per each class of videos