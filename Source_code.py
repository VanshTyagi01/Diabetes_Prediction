import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collectoin & Analysis
# Dataset -> PAMA Diabetes Dataset

# Loading the dataset to a pandas dataframe
diabetes_dataset = pd.read_csv('Diabetes_Test_dataset.csv')

# Getting the statical measures of the data
# print(diabetes_dataset.describe())
# print(diabetes_dataset['Outcome'].value_counts())
# print(diabetes_dataset.groupby('Outcome').mean())

# Separating the data and labels
X = diabetes_dataset.drop(columns= 'Outcome', axis = 1).values
Y = diabetes_dataset['Outcome'].values

# Data Stndardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

# storing standardized in X
X = standardized_data

# Spliting data into X_train, X_test, Y_train,Y_test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y, random_state = 2)
# print(X.shape, X_train.shape, X_test.shape) // data points in different sets

# Loading the Model
classifier = svm.SVC(kernel = 'linear')

# Training the support vector machine Classifier
classifier.fit(X_train,Y_train)

# Model Evaluation
#Accuracy Score of training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# print("accuracy score of training data: ", training_data_accuracy) 

#Accuracy Score of testing data
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)
# print("accuracy score of testing data: ", testing_data_accuracy)

# Making a Predictive System
# input_data = [6,148,72,35,0,33.6,0.627,50]

print("Enter the following Details as per test report->")
preg = float(input("No of Preganancies(else 0): "))
gulcose = float(input("Enter Gulcose Level (mg/dL): "))
bp = float(input("Enter Blood Pressure (mm Hg): "))
st = float(input("Enter Skin Thickness (mm): "))
insulin = float(input("Enter Insulin: "))
bmi = float(input("Enter the BMI: "))
dpf = float(input("Enter DiabetesPedigreeFunction: "))
age = float(input("Enter Your Age (Years): "))
input_data = [preg,gulcose,bp,st,insulin,bmi,dpf,age]

# changing input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# print(input_data_reshaped)
# standardized the input data
std_data = scaler.transform(input_data_reshaped)
# print(std_data)

prediction = classifier.predict(std_data)
# print(prediction)

if(prediction[0] == 0):
    print("The person is not Diabitic")
else:
    print("The person in Diabitic ")
