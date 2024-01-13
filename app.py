from flask import Flask, render_template, request
import pickle
import pandas as pd
import random

app = Flask(__name__)

# Load the saved model
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Load the dataset
# dataset_path = 'cleaned.csv' 
# df = pd.read_csv(dataset_path)

# Load the column names used during training
model_training_columns = ['Accident', 'Belts', 'Personal Injury', 'Property Damage',
                          'Alcohol', 'Violation Type', 'Race', 'Gender', 'Driver City',
                          'Driver State', 'DL State', 'Arrest Type', 'day', 'month', 'year',
                          'hour', 'SubAgency_encoded', 'Neighborhood']
# Get unique encoded values for categorical features from encodings.txt
# Get unique encoded values for categorical features from encodings.txt
unique_values = {}
encodings_file = 'encodings.txt'

with open(encodings_file, 'r') as file:
    lines = file.readlines()

current_category = None

for line in lines:
    parts = line.strip().split(', ')
    
    if len(parts) > 1:
        value, encoding = parts[0], parts[1]
        if current_category not in unique_values:
            unique_values[current_category] = []
        unique_values[current_category].append((value, encoding))
    else:
        # This line contains the category
        current_category = parts[0]

# Now, unique_values is a dictionary where each key is a category and the associated values are tuples of (value, encoding)
categorical_columns = ['Race',
    'Violation Type',
    'Driver City',
    'DL State',
    'Arrest Type',
    'SubAgency_encoded',
    'Gender',
    'Driver State',
    'Neighborhood'
    ]

# Home route
@app.route('/')
def home():
    return render_template('index.html', unique_values=unique_values)


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        input_data = {}

        for column in model_training_columns:
            if column not in categorical_columns:
                input_data[column] = [int(request.form.get(column))]
            else:
                input_data[column] = [request.form.get(column)]

        # Map input values to their encodings
        input_encoded = {}
        for column, values in unique_values.items():
            input_value = input_data[column][0]
            encoding = int(input_value.split(',')[1][2:-2])
            if encoding is not None:
                input_encoded[column] = int(encoding)
            else:
                input_encoded[column] = None

        for element in input_encoded.keys():
            input_data[element] = input_encoded[element]
        
        for element in input_data.keys():
            try:
                input_data[element] = int(input_data[element])
            except:
                pass
        input_data['Neighborhood'] = random.randint(1, 5)
        # Create a DataFrame with the input encoded data
        input_df = pd.DataFrame(input_data, index=[0])
        # Make predictions
        prediction = loaded_model.predict(input_df)
        

        return render_template('index.html', prediction=str(prediction[0]), unique_values=unique_values)
    return render_template('index.html', unique_values=unique_values)


if __name__ == '__main__':
    app.run(debug=True)
