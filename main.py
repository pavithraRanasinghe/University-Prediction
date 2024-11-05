from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('university_prediction_model.pkl')

# Load training columns to maintain the correct feature order
train_columns = joblib.load('train_columns.pkl')

# Load the dataset to map programs to universities
df = pd.read_csv('synthetic_university_dataset.csv')
program_university_mapping = df.groupby('Recommended_Program')['University'].unique().to_dict()

@app.route('/predict', methods=['POST'])
def predict():
    # Parse input JSON
    user_input = request.get_json()
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical variables and maintain column order
    input_encoded = pd.get_dummies(input_df, columns=['Interests', 'Strengths', 'Career_Goals', 'Highest_Qualification', 'Completed_OL', 'Completed_AL'])
    input_encoded = input_encoded.reindex(columns=train_columns, fill_value=0)
    
    # Get model predictions
    y_pred_probs = model.predict_proba(input_encoded)
    top_indices = y_pred_probs[0].argsort()[-10:][::-1]
    top_recommendations = [model.classes_[index] for index in top_indices]
    
    # Fetch associated universities for each predicted program
    top_recommendations_with_universities = [
        {
            "Program": program,
            "Universities": program_university_mapping.get(program, []).tolist()  # Convert ndarray to list
        }
        for program in top_recommendations
    ]
    
    # Return the top 10 recommendations with universities
    return jsonify({"Top 10 Recommended Programs and Universities": top_recommendations_with_universities})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
