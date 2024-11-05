import pandas as pd
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Create synthetic dataset with over 100 records
data = {
    "Interests": [],
    "Strengths": [],
    "Career_Goals": [],
    "Completed_OL": [],
    "Completed_AL": [],
    "Highest_Qualification": [],
    "Recommended_Program": []
}

interests = ["Science", "Commerce", "Arts", "Technology"]
strengths = ["Mathematics", "Accounting", "Literature", "Physics", "Biology", "History", "Economics", "Computer Science", "Chemistry", "Engineering"]
career_goals = ["Engineer", "Accountant", "Teacher", "Software Developer", "Doctor", "Historian", "Economist", "Data Scientist", "Researcher"]
qualifications = ["O/L", "High School"]
recommended_programs = [
    "BSc Engineering", "BCom Accounting", "BA Education", "BSc Computer Science", "MBBS",
    "BA History", "BSc Economics", "BSc Data Science", "BSc Chemistry"
]

# Generate 100 records
for _ in range(100):
    data["Interests"].append(random.choice(interests))
    data["Strengths"].append(random.choice(strengths))
    data["Career_Goals"].append(random.choice(career_goals))
    data["Completed_OL"].append("Yes")
    data["Completed_AL"].append(random.choice(["Yes", "No"]))
    data["Highest_Qualification"].append(random.choice(qualifications))
    data["Recommended_Program"].append(random.choice(recommended_programs))

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("synthetic_university_dataset.csv", index=False)
print("Dataset created successfully!")
