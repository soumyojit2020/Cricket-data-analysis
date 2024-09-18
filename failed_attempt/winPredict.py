import pandas as pd
import joblib
import pickle
import numpy as np

def predict_win_probability(gender, team1, team2, teams_type, match_type, city, toss_winner, toss_decision, method=''):
    try:
        # Load the trained model, encoder, and feature names
        model = joblib.load('cricket_win_predictor.pkl')
        encoder = joblib.load('encoder.pkl')
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)

        print(f"Loaded feature names: {feature_names}")

        # Create input data frame
        input_data = pd.DataFrame({
            'gender': [gender],
            'team_involved_one': [team1],
            'team_involved_two': [team2],
            'teams_type': [teams_type],
            'match_type': [match_type],
            'city': [city],
            'toss_winner': [toss_winner],
            'toss_decision': [toss_decision],
            'method': [method] if method else ['']
        })

        # Ensure all categorical features are strings
        categorical_features = ['gender', 'team_involved_one', 'team_involved_two', 'teams_type', 'match_type', 'city', 'toss_winner', 'toss_decision', 'method']
        input_data[categorical_features] = input_data[categorical_features].astype(str)

        print("Input data:")
        print(input_data)

        # Encode the input data
        encoded_input_data = encoder.transform(input_data[categorical_features]).toarray()
        encoded_feature_names = encoder.get_feature_names_out(categorical_features)
        encoded_input_df = pd.DataFrame(encoded_input_data, columns=encoded_feature_names)
        print('encode get features names out:',encoder.get_feature_names_out())
        # Create a DataFrame with all expected features, filling in missing ones with zeros
        final_input = pd.DataFrame(0, index=range(1), columns=feature_names)
        for col in encoded_input_df.columns:
            if col in feature_names:
                final_input[col] = encoded_input_df[col]

        print("Final input features:")
        print(final_input.columns)

        # Make prediction
        prediction = model.predict_proba(final_input)

        return prediction

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage
if __name__ == "__main__":
    team1 = 'Sri Lanka'
    team2 = 'India'
    prediction = predict_win_probability('male', team1, team2, 'international', 'T20', 'Colombo', team1, 'field', '')
    if prediction is not None:
        print(f'Probability of {team1} winning: {prediction[0][0]}')
        print(f'Probability of {team2} winning: {prediction[0][1]}')
    else:
        print("Prediction failed.")