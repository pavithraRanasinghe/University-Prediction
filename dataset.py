import pandas as pd
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Create synthetic dataset with over 200 records (to ensure multiple universities per program)
data = {
    "Interests": [],
    "Strengths": [],
    "Career_Goals": [],
    "Completed_OL": [],
    "Completed_AL": [],
    "Highest_Qualification": [],
    "Recommended_Program": [],
    "University": []
}

interests = ["Science", "Commerce", "Arts", "Technology"]
strengths = ["Mathematics", "Accounting", "Literature", "Physics", "Biology", "History", "Economics", "Computer Science", "Chemistry", "Engineering"]
career_goals = ["Engineer", "Accountant", "Teacher", "Software Developer", "Doctor", "Historian", "Economist", "Data Scientist", "Researcher"]
qualifications = ["O/L", "High School"]
recommended_programs = [
    "BSc Engineering", "BCom Accounting", "BA Education", "BSc Computer Science", "MBBS",
    "BA History", "BSc Economics", "BSc Data Science", "BSc Chemistry"
]
universities = [
    "University of Colombo", "University of Sri Jayewardenepura", "University of Kelaniya", "University of Moratuwa",
    "Rajarata University", "Wayamba University", "University of Peradeniya", "Eastern University", "South Eastern University"
]

# Generate 200 records, ensuring multiple universities per program
for _ in range(200):
    data["Interests"].append(random.choice(interests))
    data["Strengths"].append(random.choice(strengths))
    data["Career_Goals"].append(random.choice(career_goals))
    data["Completed_OL"].append("Yes")
    data["Completed_AL"].append(random.choice(["Yes", "No"]))
    data["Highest_Qualification"].append(random.choice(qualifications))
    program = random.choice(recommended_programs)
    university = random.choice(universities)
    data["Recommended_Program"].append(program)
    data["University"].append(university)

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("synthetic_university_dataset.csv", index=False)
print("Dataset created successfully!")
