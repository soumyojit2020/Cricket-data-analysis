import numpy as np
import joblib

# Load the model, feature columns, and mappings
model = joblib.load('cricket_model.joblib')
feature_columns = joblib.load('feature_columns.joblib')
mappings = joblib.load('mappings.joblib')

def predict_win_probability(gender, teams_type, match_type, team1, team2, city, toss_winner, toss_decision):
    # Encode input features
    input_data = np.zeros(len(feature_columns))
    
    input_data[feature_columns.index('gender_Male')] = mappings['gender'].get(gender.lower(), 0)
    input_data[feature_columns.index('teams_type_international')] = mappings['teams_type'].get(teams_type.lower(), 0)
    input_data[feature_columns.index('match_type_freq')] = 0.5  # You might want to implement a better way to handle this
    input_data[feature_columns.index('city_freq')] = 0.5  # You might want to implement a better way to handle this
    input_data[feature_columns.index('toss_decision_field')] = mappings['toss_decision'].get(toss_decision.lower(), 0)
    input_data[feature_columns.index('team_involved_one_encoded')] = mappings['reverse_team'].get(team1, -1)
    input_data[feature_columns.index('team_involved_two_encoded')] = mappings['reverse_team'].get(team2, -1)
    input_data[feature_columns.index('toss_winner_encoded')] = mappings['reverse_team'].get(toss_winner, -1)

    # Make prediction
    proba = model.predict_proba(input_data.reshape(1, -1))[0]

    # Map probabilities to teams
    team1_idx = mappings['reverse_team'].get(team1, -1)
    team2_idx = mappings['reverse_team'].get(team2, -1)

    prob1 = proba[team1_idx] if team1_idx != -1 and team1_idx < len(proba) else 0
    prob2 = proba[team2_idx] if team2_idx != -1 and team2_idx < len(proba) else 0

    # Normalize probabilities
    total_prob = prob1 + prob2
    if total_prob > 0:
        prob1 = (prob1 / total_prob) * 100
        prob2 = (prob2 / total_prob) * 100
    else:
        prob1 = prob2 = 50  # If both probabilities are 0, assign 50% to each

    return prob1, prob2

def get_input_and_predict():
    print("Enter match details:")
    gender = input("Gender (male/female): ")
    teams_type = input("Teams type (international/club): ")
    match_type = input("Match type: ")
    team1 = input("Team 1: ")
    team2 = input("Team 2: ")
    city = input("City: ")
    toss_winner = input("Toss winner: ")
    toss_decision = input("Toss decision (bat/field): ")

    prob1, prob2 = predict_win_probability(gender, teams_type, match_type, team1, team2, city, toss_winner, toss_decision)

    print(f"\n{team1} win probability: {prob1:.2f}%")
    print(f"{team2} win probability: {prob2:.2f}%")

if __name__ == "__main__":
    get_input_and_predict()