import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load job requirements and candidate data
job_requirements = pd.read_csv("./job.csv")
candidates = pd.read_csv("./candidates.csv")

# Preprocess data
vectorizer = TfidfVectorizer()
job_req_vector = vectorizer.fit_transform(job_requirements["skills"])
candidate_vector = vectorizer.transform(candidates["resume"])

# Create feature matrix
X = pd.concat([job_req_vector, candidate_vector], axis=1)
y = candidates["suitable"]  # assume 'suitable' column indicates suitability

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model performance
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Predict suitability for each candidate
candidates["suitability_score"] = model.predict(candidate_vector)
