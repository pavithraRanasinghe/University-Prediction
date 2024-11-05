import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib

# Step 1: Load Dataset
df = pd.read_csv('synthetic_university_dataset.csv')

# Step 2: Data Preprocessing
# Encode categorical variables
categorical_columns = ['Interests', 'Strengths', 'Career_Goals', 'Highest_Qualification', 'Completed_OL', 'Completed_AL']
df_encoded = pd.get_dummies(df, columns=categorical_columns)

# Save the training columns for use in the API
train_columns = df_encoded.drop(columns=['Recommended_Program', 'University']).columns
joblib.dump(train_columns, 'train_columns.pkl')

# Split the dataset into features and target
X = df_encoded.drop(columns=['Recommended_Program', 'University'])
y = df_encoded[['Recommended_Program', 'University']]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train['Recommended_Program'])

# Step 4: Model Evaluation and Multi-Label Prediction
y_pred_probs = model.predict_proba(X_test)

# Get top 10 predictions for each sample using probabilities
top_n = 10
predictions = []

for prob in y_pred_probs:
    top_indices = prob.argsort()[-top_n:][::-1]
    top_classes = [model.classes_[index] for index in top_indices]
    predictions.append(top_classes)

# Fetch the associated universities for each predicted program
predictions_with_universities = []
for pred_list in predictions:
    university_list = [y_train[y_train['Recommended_Program'] == program]['University'].iloc[0] for program in pred_list]
    predictions_with_universities.append(list(zip(pred_list, university_list)))

# Step 5: Cosine Similarity for Ranking
# Normalize the test data
X_test_normalized = normalize(X_test)

# Compute cosine similarity between each test sample and all training samples
top_n_cosine_predictions = []
for i in range(len(X_test_normalized)):
    similarities = cosine_similarity([X_test_normalized[i]], X_train)[0]
    top_similar_indices = similarities.argsort()[-top_n:][::-1]
    top_similar_programs = y_train.iloc[top_similar_indices][['Recommended_Program', 'University']].to_dict(orient='records')
    top_n_cosine_predictions.append(top_similar_programs)

# Example: Print the top 10 recommended programs and universities for the first few test samples based on cosine similarity
for i, pred in enumerate(predictions_with_universities[:5]):
    print(f"Sample {i+1} Top {top_n} Recommended Programs and Universities: {pred}")

# Evaluate accuracy based on top prediction from Random Forest
y_pred_top_1 = [pred[0][0] for pred in predictions_with_universities]
accuracy = accuracy_score(y_test['Recommended_Program'], y_pred_top_1)
print("Model Accuracy (Top 1 Prediction):", accuracy)
print("Classification Report (Top 1 Prediction):\n", classification_report(y_test['Recommended_Program'], y_pred_top_1))

# Step 6: Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train['Recommended_Program'])
print("Best Parameters:", grid_search.best_params_)

# Save the trained model to a file
joblib.dump(model, 'university_prediction_model.pkl')
print("Model saved successfully!")
