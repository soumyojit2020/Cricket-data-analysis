import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

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
df['team1_avg_runs'] = df['team_involved_one'].map(lambda x: team_stats[x]['runs_scored'] / team_stats[x]['matches'] if team_stats[x]['matches'] > 0 else 0)
df['team2_avg_runs'] = df['team_involved_two'].map(lambda x: team_stats[x]['runs_scored'] / team_stats[x]['matches'] if team_stats[x]['matches'] > 0 else 0)

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
numerical_columns = ['runs', 'innings', 'wickets', 'team1_win_ratio', 'team2_win_ratio', 'team1_avg_runs', 'team2_avg_runs']
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# Prepare features and target
features = categorical_columns + numerical_columns + ['home_advantage', 'toss_winner_team1']
X = df[features]
y = (df['winner'] == df['team_involved_one']).astype(float)  # 1 if team_involved_one wins, 0 otherwise

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Function to predict winner
def predict_winner(team1, team2, match_type, city, toss_winner, toss_decision):
    # Prepare input data
    input_data = pd.DataFrame({
        'teams_type': ['international'],
        'match_type': [match_type],
        'team_involved_one': [team1],
        'team_involved_two': [team2],
        'city': [city],
        'toss_winner': [toss_winner],
        'toss_decision': [toss_decision],
        'runs': [0],
        'innings': [0],
        'wickets': [0],
        'team1_win_ratio': [team_stats[team1]['wins'] / team_stats[team1]['matches'] if team_stats[team1]['matches'] > 0 else 0],
        'team2_win_ratio': [team_stats[team2]['wins'] / team_stats[team2]['matches'] if team_stats[team2]['matches'] > 0 else 0],
        'team1_avg_runs': [team_stats[team1]['runs_scored'] / team_stats[team1]['matches'] if team_stats[team1]['matches'] > 0 else 0],
        'team2_avg_runs': [team_stats[team2]['runs_scored'] / team_stats[team2]['matches'] if team_stats[team2]['matches'] > 0 else 0],
        'home_advantage': [1 if team1 == city else 0],
        'toss_winner_team1': [1 if toss_winner == team1 else 0]
    })
    
    # Encode categorical variables
    for col in categorical_columns:
        try:
            input_data[col] = le_dict[col].transform(input_data[col].astype(str))
        except ValueError:
            # If the category is not in the training data, assign a new unique integer
            new_category = max(le_dict[col].classes_) + 1
            le_dict[col].classes_ = np.append(le_dict[col].classes_, input_data[col].iloc[0])
            input_data[col] = new_category
    
    # Normalize numerical features
    input_data[numerical_columns] = scaler.transform(input_data[numerical_columns])
    
    # Make prediction
    prob_team1_wins = rf_model.predict(input_data[features])[0]
    
    return prob_team1_wins

# Example usage
team1 = 'Croatia'
team2 = 'Spain'
match_type = 'T20'
city = 'Zagreb'
toss_winner = 'Croatia'
toss_decision = 'field'

prob_team1_wins = predict_winner(team1, team2, match_type, city, toss_winner, toss_decision)
print(f"Probability of {team1} winning: {prob_team1_wins:.2f}")
print(f"Probability of {team2} winning: {1 - prob_team1_wins:.2f}")