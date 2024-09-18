import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the data
df = pd.read_csv('matches_data.csv')

# Calculate win ratios for each team
def calculate_win_ratios(df):
    team_win_ratios = {}
    for team in pd.concat([df['team_involved_one'], df['team_involved_two']]).unique():
        team_matches = df[(df['team_involved_one'] == team) | (df['team_involved_two'] == team)]
        team_wins = team_matches[team_matches['winner'] == team]
        team_win_ratios[team] = len(team_wins) / len(team_matches) if len(team_matches) > 0 else 0.5
    return team_win_ratios

team_win_ratios = calculate_win_ratios(df)

# Create a robust encoding function
def robust_encode(series):
    unique_values = series.unique()
    return {val: idx for idx, val in enumerate(unique_values)}

# Encode categorical variables
teams_type_encoding = robust_encode(df['teams_type'])
match_type_encoding = robust_encode(df['match_type'])
city_encoding = robust_encode(df['city'])
toss_decision_encoding = robust_encode(df['toss_decision'])

# Create features
def create_features(row):
    features = [
        team_win_ratios[row['team_involved_one']],
        team_win_ratios[row['team_involved_two']],
        teams_type_encoding[row['teams_type']],
        match_type_encoding[row['match_type']],
        city_encoding[row['city']],
        1 if row['toss_winner'] == row['team_involved_one'] else 0,
        toss_decision_encoding[row['toss_decision']]
    ]
    return features

X = df.apply(create_features, axis=1, result_type='expand')
X.columns = ['team1_win_ratio', 'team2_win_ratio', 'teams_type', 'match_type', 'city', 'team1_won_toss', 'toss_decision']

# Define target variable
y = (df['winner'] == df['team_involved_one']).astype(int)

# Print class distribution
print("Class distribution:", np.bincount(y))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Function to predict winner probabilities
def predict_winner_proba(team1, team2, teams_type, match_type, city, toss_winner, toss_decision):
    team1_ratio = team_win_ratios.get(team1, 0.5)
    team2_ratio = team_win_ratios.get(team2, 0.5)
    
    try:
        teams_type_encoded = teams_type_encoding.get(teams_type, -1)
        match_type_encoded = match_type_encoding.get(match_type, -1)
        city_encoded = city_encoding.get(city, -1)
        toss_decision_encoded = toss_decision_encoding.get(toss_decision, -1)
    except KeyError as e:
        print(f"Warning: Unknown category encountered: {e}")
        return 0.5, 0.5  # Return equal probabilities if we encounter an unknown category
    
    features = [
        team1_ratio,
        team2_ratio,
        teams_type_encoded,
        match_type_encoded,
        city_encoded,
        1 if toss_winner == team1 else 0,
        toss_decision_encoded
    ]
    
    if -1 in features:
        print("Warning: Unknown category encountered. Prediction may be less accurate.")
    
    proba = model.predict_proba([features])[0]
    return proba[1], proba[0]  # probability of team1 winning, probability of team2 winning

# Example usage
def predict_match(team1, team2, teams_type, match_type, city, toss_winner, toss_decision):
    prob_team1_wins, prob_team2_wins = predict_winner_proba(team1, team2, teams_type, match_type, city, toss_winner, toss_decision)
    print(f"Probability of {team1} winning: {prob_team1_wins:.2f}")
    print(f"Probability of {team2} winning: {prob_team2_wins:.2f}")

# Predict some example matches
print("\nPredictions:")
predict_match('Sri Lanka', 'India', 'international', 'T20', 'Colombo', 'India', 'field')