import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'matches_data.csv'
df = pd.read_csv(file_path)

# # Display the first few rows of the dataset
# print("First 5 rows of the dataset:")
# print(df.head())

# # Get an overview of the dataset (column names, non-null counts, data types)
# print("\nDataset Info:")
# print(df.info())

# # Check for missing values
# print("\nMissing Values in Each Column:")
# print(df.isnull().sum())

# # Statistical summary of numerical columns
# print("\nStatistical Summary of Numerical Columns:")
# print(df.describe())

# Drop the specified columns
df = df.drop(columns=['match_id', 'runs', 'innings', 'wickets', 'method'])

# Convert 'start_date' to datetime format
df['start_date'] = pd.to_datetime(df['start_date'])

# Extract 'day', 'month', and 'year' from 'start_date'
df['day'] = df['start_date'].dt.day
df['month'] = df['start_date'].dt.month
df['year'] = df['start_date'].dt.year

# Drop the original 'start_date' column if not needed anymore
df = df.drop(columns=['start_date'])

# Fill missing values in the 'winner' column with corresponding values from the 'result' column
df['winner'] = df['winner'].fillna(df['result'])

# Drop the 'result' column
df = df.drop(columns=['result'])

# # Verify the changes
# print("Columns after dropping and filling missing values:")
# print(df.info())

# print("\nCheck for remaining missing values:")
# print(df.isnull().sum())

# # Display the first few rows of the dataset after processing
# print("First 5 rows of the dataset after processing:")
# print(df.head())

#start encoding
# Step 1: One-Hot Encoding for 'gender', 'teams_type', and 'toss_decision'
df = pd.get_dummies(df, columns=['gender', 'teams_type', 'toss_decision'], drop_first=True)

# Step 2: Frequency Encoding for 'match_type' and 'city'
for col in ['match_type', 'city']:
    freq_encoding = df[col].value_counts(normalize=True)
    df[col + '_freq'] = df[col].map(freq_encoding)
    df = df.drop(columns=[col])

team_names = pd.concat([df['team_involved_one'], df['team_involved_two']]).unique()
team_encoder = {team: idx for idx, team in enumerate(team_names)}

special_case_mapping = {
    'tie': len(team_encoder),
    'no result': len(team_encoder) + 1,
    'draw': len(team_encoder) + 2
}

# Apply the mappings to 'team_involved_one', 'team_involved_two', and 'toss_winner'
df['team_involved_one_encoded'] = df['team_involved_one'].map(team_encoder)
df['team_involved_two_encoded'] = df['team_involved_two'].map(team_encoder)
df['toss_winner_encoded'] = df['toss_winner'].map(team_encoder)

# Map winner with both team names and special cases
# Assign fixed values to special cases
df['winner_encoded'] = df['winner'].apply(lambda x: special_case_mapping.get(x, team_encoder.get(x, pd.NA)))

# Create a LabelEncoder
le = LabelEncoder()

# Fit and transform the 'winner_encoded' column
df['winner_encoded'] = le.fit_transform(df['winner_encoded'].astype(str))

# Drop original columns after encoding
df = df.drop(columns=['team_involved_one', 'team_involved_two', 'toss_winner', 'winner'])

# Ensure 'winner_encoded' is numeric
df['winner_encoded'] = pd.to_numeric(df['winner_encoded'], errors='coerce')

# Verify the final structure of the dataset
print("Final DataFrame structure:")
print(df.info())

print("\nFirst 5 rows of the dataset after encoding:")
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('encoded_data.csv', index=False)