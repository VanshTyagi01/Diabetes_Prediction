# Description

This project is a machine learning-based predictive system designed to diagnose diabetes using the PIMA Diabetes Dataset. The system leverages a Support Vector Machine (SVM) classifier to predict whether a person is diabetic based on various health metrics such as the number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.

Procedure -------------<br>

1-Data Collection and Analysis:<br>
    &emsp;&emsp;The dataset is loaded into a pandas DataFrame for easy manipulation and analysis.<br>
    &emsp;&emsp;Statistical measures and data distribution are analyzed to understand the dataset better.<br>
    
2-Data Preprocessing:<br>
    &emsp;&emsp;The dataset is split into features (X) and labels (Y).<br>
    &emsp;&emsp;Standardization is applied to the features to ensure that they have a mean of 0 and a standard 
    deviation of 1, which helps in improving the performance of the SVM classifier.<br>
    
3-Model Training:<br>
    &emsp;&emsp;The data is split into training and testing sets to evaluate the model’s performance.<br>
    &emsp;&emsp;An SVM classifier with a linear kernel is trained on the training data.<br>
    
4-Model Evaluation:<br>
    &emsp;&emsp;The accuracy of the model is evaluated on both the training and testing datasets to ensure it 
    generalizes well to unseen data.<br>
    
5-Predictive System:<br>
    &emsp;&emsp;The system takes user input for various health metrics and standardizes this input.<br>
    &emsp;&emsp;The trained SVM model then predicts whether the person is diabetic based on the input data.<br>
