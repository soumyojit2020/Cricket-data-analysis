#simple model based on win ratio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

# Create features based on team win ratios
X = df.apply(lambda row: [
    team_win_ratios[row['team_involved_one']], 
    team_win_ratios[row['team_involved_two']]
], axis=1, result_type='expand')
X.columns = ['team1_win_ratio', 'team2_win_ratio']

# Define target variable
y = (df['winner'] == df['team_involved_one']).astype(int)

# Print class distribution
print("Class distribution:", np.bincount(y))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a simple model
def predict_winner(team1_ratio, team2_ratio):
    return 1 if team1_ratio > team2_ratio else 0

# Make predictions
y_pred = X_test.apply(lambda row: predict_winner(row['team1_win_ratio'], row['team2_win_ratio']), axis=1)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Function to predict winner probabilities
def predict_winner_proba(team1, team2):
    team1_ratio = team_win_ratios.get(team1, 0.5)
    team2_ratio = team_win_ratios.get(team2, 0.5)
    total = team1_ratio + team2_ratio
    prob_team1_wins = team1_ratio / total
    prob_team2_wins = team2_ratio / total
    return prob_team1_wins, prob_team2_wins

# Example usage
def predict_match(team1, team2):
    prob_team1_wins, prob_team2_wins = predict_winner_proba(team1, team2)
    print(f"Probability of {team1} winning: {prob_team1_wins:.2f}")
    print(f"Probability of {team2} winning: {prob_team2_wins:.2f}")

# Predict some example matches
print("\nPredictions:")
predict_match('Sri Lanka', 'India')
predict_match('Australia', 'England')
predict_match('South Africa', 'New Zealand')

# # Function to get top N teams based on win ratio
# def get_top_teams(n=10):
#     sorted_teams = sorted(team_win_ratios.items(), key=lambda x: x[1], reverse=True)
#     return sorted_teams[:n]

# # Print top 10 teams
# print("\nTop 10 teams by win ratio:")
# for team, ratio in get_top_teams(10):
#     print(f"{team}: {ratio:.2f}")