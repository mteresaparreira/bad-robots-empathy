## SuperBAD: Repository Information

This project has been performed in the Future Automation Research Laboratory (FARLabs) at Cornell Tech.  

Advisor: Dr. Wendy Ju  
Doctoral Researcher: Maria Teresa Bento Parreira  
Masters Researcher: Sukruth Gowdru Lingaraju

This repository consists of the code used in the SuperBAD project.

The `code` directory consists of sub-directories of the machine learning models that are used in the implementations of various models for multi-label classification of participant reactions to the below classes:

1. Control
1. Failure - Human
1. Failure - Robot

The data for these models are extracted from the [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace) library - where the responseVideos of the participants to the stimulusVideos are fed and the required features are extracted.

These features are classified accordingly to the class they belong to (based on the stimulusVideo).

The feature values for all the videos for all participants are then appended together - which act as the final dataset for the models.

For the participant response dataset - please contact --add relevant name--.