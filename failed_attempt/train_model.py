import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import time
from datetime import timedelta

def timer(start_time=None):
    if not start_time:
        start_time = time.time()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(time.time() - start_time, 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

total_start_time = time.time()

# Load the encoded data
print("Loading data...")
start_time = timer()
df = pd.read_csv('encoded_data.csv')
timer(start_time)

# Prepare features and target
print("Preparing features and target...")
start_time = timer()
feature_columns = ['gender_male', 'teams_type_international', 'match_type_freq', 'city_freq', 
                   'toss_decision_field', 'team_involved_one_encoded', 'team_involved_two_encoded', 
                   'toss_winner_encoded']
X = df[feature_columns]
y = df['winner_encoded']

# Create a custom mapping for classes
unique_classes = sorted(y.unique())
class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
y = y.map(class_mapping)

num_classes = len(class_mapping)
print(f"Number of unique classes after mapping: {num_classes}")
timer(start_time)

# Split the data
print("Splitting data...")
start_time = timer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
timer(start_time)

print("Class distribution in training data:")
print(y_train.value_counts().sort_index())
print(f"Number of unique classes in training data: {len(np.unique(y_train))}")

# Define model
print("Defining model...")
start_time = timer()
model = XGBClassifier(
    objective='multi:softprob',
    num_class=num_classes,
    use_label_encoder=False,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
timer(start_time)

# Train the model
print("Training model...")
start_time = timer()
model.fit(X_train, y_train)
timer(start_time)

# Evaluate the model
print("Evaluating model...")
start_time = timer()
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
timer(start_time)

# Save the model
print("Saving model...")
start_time = timer()
joblib.dump(model, 'cricket_model.joblib')
joblib.dump(feature_columns, 'feature_columns.joblib')
joblib.dump(class_mapping, 'class_mapping.joblib')
timer(start_time)

print("Model, feature columns, and class mapping saved successfully.")

# Total time taken
total_time = time.time() - total_start_time
print(f"\nTotal time taken: {str(timedelta(seconds=total_time))}")