import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        """ Load dataset from CSV file. """
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column):
        """ Separate features and target variable. """
        self.output_df = self.data[target_column]
        self.input_df = self.data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

        # ✅ Debugging: Print selected features
        print("Features used for training:", self.input_df.columns.tolist())
        print("Number of features:", self.input_df.shape[1])

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.model = None
        self.x_train, self.x_test, self.y_train, self.y_test = [None] * 4
        self.y_predict = None
        self.create_model()
    
    def create_model(self, criterion='gini', max_depth=6):
        self.model = RandomForestClassifier(criterion=criterion, max_depth=max_depth)
        
    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size, random_state=random_state)
    
    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
    
    def evaluate_model(self):
        predictions = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, predictions)
        return accuracy
    
    def make_prediction(self):
        self.y_predict = self.model.predict(self.x_test)
    
    def create_report(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict))
    
    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

# ✅ Main Program
file_path = "Iris.csv"  # Ensure this file exists in the same directory

data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('Species')

model_handler = ModelHandler(data_handler.input_df, data_handler.output_df)
model_handler.split_data()

print("📌 Before Training")
model_handler.train_model()
print("✅ Model Accuracy:", model_handler.evaluate_model())

model_handler.make_prediction()
model_handler.create_report()

# ✅ Save the newly trained model
model_filename = 'trained_model.pkl'
model_handler.save_model(model_filename)
print(f"🎉 Model saved successfully as {model_filename}")