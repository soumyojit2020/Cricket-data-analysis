import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

# Load the data
df = pd.read_csv('matches_data.csv')

# Function to create historical performance features
def create_historical_features(df):
    team_stats = {}
    for _, row in df.iterrows():
        team1, team2 = row['team_involved_one'], row['team_involved_two']
        winner = row['winner']
        
        for team in [team1, team2]:
            if team not in team_stats:
                team_stats[team] = {'matches': 0, 'wins': 0, 'runs_scored': 0, 'runs_conceded': 0}
            
            team_stats[team]['matches'] += 1
            if team == winner:
                team_stats[team]['wins'] += 1
            
            if team == team1:
                team_stats[team]['runs_scored'] += row['runs'] if not pd.isna(row['runs']) else 0
            else:
                team_stats[team]['runs_conceded'] += row['runs'] if not pd.isna(row['runs']) else 0
    
    return team_stats

# Create historical features
team_stats = create_historical_features(df)

# Add historical features to the dataframe
df['team1_win_ratio'] = df['team_involved_one'].map(lambda x: team_stats[x]['wins'] / team_stats[x]['matches'] if team_stats[x]['matches'] > 0 else 0)
df['team2_win_ratio'] = df['team_involved_two'].map(lambda x: team_stats[x]['wins'] / team_stats[x]['matches'] if team_stats[x]['matches'] > 0 else 0)

# Encode categorical variables
le_dict = {}
categorical_columns = ['teams_type', 'match_type', 'team_involved_one', 'team_involved_two', 'city', 'toss_winner', 'toss_decision']
for col in categorical_columns:
    le_dict[col] = LabelEncoder()
    df[col] = le_dict[col].fit_transform(df[col].astype(str))

# Create binary features
df['home_advantage'] = (df['team_involved_one'] == df['city']).astype(int)
df['toss_winner_team1'] = (df['toss_winner'] == df['team_involved_one']).astype(int)

# Handle missing data
imputer = SimpleImputer(strategy='mean')
df[['runs', 'innings', 'wickets']] = imputer.fit_transform(df[['runs', 'innings', 'wickets']])

# Normalize numerical features
scaler = StandardScaler()
numerical_columns = ['runs', 'innings', 'wickets', 'team1_win_ratio', 'team2_win_ratio']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Prepare features and target
features = categorical_columns + numerical_columns + ['home_advantage', 'toss_winner_team1']
X = df[features]
y = (df['winner'] == df['team_involved_one']).astype(int)  # 1 if team_involved_one wins, 0 otherwise

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Ensure both classes are represented
if len(np.unique(y)) == 1:
    print("Warning: Only one class present in the target variable. Adding a dummy sample for the other class.")
    X = pd.concat([X, X.iloc[0:1]])
    y = np.concatenate([y, [1 - y[0]]])

# Train the model with the resampled data
# rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42)
rf_model.fit(X_resampled, y_resampled)
print('rf_model:',rf_model.classes_)

from sklearn.calibration import CalibratedClassifierCV

# Calibrate the model
# Replace rf_model with calibrated_rf in your predict_winner function
calibrated_rf = CalibratedClassifierCV(rf_model, cv=5, method='sigmoid')
calibrated_rf.fit(X_resampled, y_resampled)
print('calibrated_rf:',calibrated_rf.classes_)


# Evaluate the model using cross-validation
cv_scores = cross_val_score(rf_model, X_resampled, y_resampled, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Function to predict winner
def predict_winner(team1, team2, teams_type, match_type, city, toss_winner, toss_decision):
    # Prepare input data
    input_data = pd.DataFrame({
        'teams_type': [teams_type],
        'match_type': [match_type],
        'team_involved_one': [team1],
        'team_involved_two': [team2],
        'city': [city],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'runs': [0],
        'innings': [0],
        'wickets': [0],
        'team1_win_ratio': [team_stats.get(team1, {'wins': 0, 'matches': 1})['wins'] / team_stats.get(team1, {'wins': 0, 'matches': 1})['matches']],
        'team2_win_ratio': [team_stats.get(team2, {'wins': 0, 'matches': 1})['wins'] / team_stats.get(team2, {'wins': 0, 'matches': 1})['matches']],
        'home_advantage': [1 if team1 == city else 0],
        'toss_winner_team1': [1 if toss_winner == team1 else 0]
    })
    
    # Encode categorical variables
    for col in categorical_columns:
        if col in le_dict:
            input_data[col] = input_data[col].map(lambda x: le_dict[col].transform([x])[0] if x in le_dict[col].classes_ else -1)
        else:
            input_data[col] = -1  # Use -1 for unknown categories
    
    # Normalize numerical features
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
    
    # Make prediction
    prediction = calibrated_rf.predict_proba(input_data[features])[0]
    
    if len(prediction) == 1:
        # If only one class is predicted, assume it's the probability for class 1
        prob_team1_wins = prediction[0]
        prob_team2_wins = 1 - prob_team1_wins
    else:
        prob_team1_wins = prediction[1]
        prob_team2_wins = prediction[0]
    
    return prob_team1_wins, prob_team2_wins

# Example usage
team1 = 'Sri Lanka'
team2 = 'India'
teams_type = 'international'
match_type = 'T20'
city = 'Colombo'
toss_winner = 'India'
toss_decision = 'field'

prob_team1_wins, prob_team2_wins = predict_winner(team1, team2, teams_type, match_type, city, toss_winner, toss_decision)
print(f"Probability of {team1} winning: {prob_team1_wins:.2f}")
print(f"Probability of {team2} winning: {prob_team2_wins:.2f}")