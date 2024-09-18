import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import time

def print_time_taken(start_time, task):
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken for {task}: {elapsed_time:.2f} seconds")

try:
    start_time = time.time()

    # Load the data
    print("Loading data...")
    data_start_time = time.time()
    df = pd.read_csv('matches_data.csv')
    print_time_taken(data_start_time, "loading data")

    # Handle missing values
    print("Handling missing values...")
    missing_values_start_time = time.time()
    df.fillna(0, inplace=True)
    print_time_taken(missing_values_start_time, "handling missing values")

    # Handle match_id column: converting non-integer IDs to numerical
    df['match_id'] = df['match_id'].apply(lambda x: int(x.split('_')[-1]) if isinstance(x, str) and 'wi_' in x else int(x))

    # Convert start_date to datetime and extract useful features
    print("Processing start_date column...")
    start_date_start_time = time.time()
    df['start_date'] = pd.to_datetime(df['start_date'], format='%m/%d/%Y')
    df['year'] = df['start_date'].dt.year
    df['month'] = df['start_date'].dt.month
    df['day'] = df['start_date'].dt.day
    df.drop(columns=['start_date'], inplace=True)
    print_time_taken(start_date_start_time, "processing start_date column")

    # Ensure all categorical features are strings
    categorical_features = ['gender', 'teams_type', 'match_type', 'team_involved_one', 'team_involved_two', 'city', 'toss_winner', 'toss_decision', 'method']
    df[categorical_features] = df[categorical_features].astype(str)

    # Ensure target variable 'winner' is a string
    df['winner'] = df['winner'].astype(str)

    # Drop classes with too few samples
    print("Dropping classes with too few samples...")
    drop_classes_start_time = time.time()
    min_samples = 2  # Minimum number of samples required to include a class
    class_counts = df['winner'].value_counts()
    df = df[df['winner'].isin(class_counts[class_counts >= min_samples].index)]
    print_time_taken(drop_classes_start_time, "dropping classes with too few samples")

    # Feature Engineering: Convert categorical features to numerical values
    print("Encoding categorical features...")
    encoding_start_time = time.time()
    encoder = OneHotEncoder()
    encoded_features = encoder.fit_transform(df[categorical_features]).toarray()
    feature_names = encoder.get_feature_names_out(categorical_features)
    encoded_df = pd.DataFrame(encoded_features, columns=feature_names)
    df = pd.concat([df.drop(columns=categorical_features), encoded_df], axis=1)
    print_time_taken(encoding_start_time, "encoding categorical features")

    # Define the target variable and features
    X = df.drop(columns=['result', 'winner'])
    y = df['winner'].astype(str)  # Ensure y is consistently a string

    # Ensure no NaN values in the feature matrix
    if X.isnull().values.any():
        print("Handling NaN values in feature matrix...")
        nan_values_start_time = time.time()
        X.fillna(0, inplace=True)
        print_time_taken(nan_values_start_time, "handling NaN values in feature matrix")

    # Handle class imbalance
    print("Handling class imbalance with SMOTE...")
    smote_start_time = time.time()
    smote = SMOTE(k_neighbors=min(min_samples-1, 5))  # Set k_neighbors to min_samples-1 or 5, whichever is smaller
    X, y = smote.fit_resample(X, y)
    print_time_taken(smote_start_time, "handling class imbalance with SMOTE")

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    split_start_time = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print_time_taken(split_start_time, "splitting data into training and testing sets")

    # Initialize the model
    print("Initializing RandomForest model...")
    model_initialization_start_time = time.time()
    model = RandomForestClassifier()
    print_time_taken(model_initialization_start_time, "initializing RandomForest model")

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Initialize GridSearchCV with StratifiedKFold
    print("Initializing GridSearchCV...")
    grid_search_initialization_start_time = time.time()
    cv = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv)
    print_time_taken(grid_search_initialization_start_time, "initializing GridSearchCV")

    # Fit the grid search to the data
    print("Fitting GridSearchCV...")
    grid_search_fit_start_time = time.time()
    grid_search.fit(X_train, y_train)
    print_time_taken(grid_search_fit_start_time, "fitting GridSearchCV")

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f'Best Parameters: {best_params}')

    # Train the final model with the best parameters
    print("Training the final model with the best parameters...")
    final_model_training_start_time = time.time()
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)
    print_time_taken(final_model_training_start_time, "training the final model with the best parameters")

    # Make predictions
    print("Making predictions...")
    prediction_start_time = time.time()
    y_pred = best_model.predict(X_test)
    print_time_taken(prediction_start_time, "making predictions")

    # Evaluate the model
    print("Evaluating the model...")
    evaluation_start_time = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print_time_taken(evaluation_start_time, "evaluating the model")

    # Save the model
    print("Saving the model and encoder...")
    save_model_start_time = time.time()
    joblib.dump(best_model, 'cricket_win_predictor.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    print_time_taken(save_model_start_time, "saving the model")

    print("Script completed successfully!")
    print_time_taken(start_time, "total script execution")

except Exception as e:
    print(f"An error occurred: {e}")

# Load the model and make predictions
def predict_win_probability(gender, team1, team2, teams_type, match_type, city, toss_winner, toss_decision, method=''):
    try:
        start_time = time.time()
        
        # Load the trained model and encoder
        print("Loading model and encoder...")
        model = joblib.load('cricket_win_predictor.pkl')
        encoder = joblib.load('encoder.pkl')
        
        load_time = time.time()
        print(f"Model and encoder loaded in {load_time - start_time:.2f} seconds")
        
        # Create input data frame
        print("Creating input data frame...")
        input_data = pd.DataFrame({
            'gender': [gender],
            'team_involved_one': [team1],
            'team_involved_two': [team2],
            'teams_type': [teams_type],
            'match_type': [match_type],
            'city': [city],
            'toss_winner': [toss_winner],
            'toss_decision': [toss_decision],
            'method': [method] if method else ['']  # Handle blank values in the method field
        })

        # Ensure all categorical features are strings
        print("Ensuring categorical features are strings...")
        categorical_features = ['gender', 'team_involved_one', 'team_involved_two', 'teams_type', 'match_type', 'city', 'toss_winner', 'toss_decision', 'method']
        input_data[categorical_features] = input_data[categorical_features].astype(str)

        # Encode the input data
        print("Encoding input data...")
        encoded_input_data = encoder.transform(input_data).toarray()
        feature_names = encoder.get_feature_names_out(categorical_features)
        encoded_input_df = pd.DataFrame(encoded_input_data, columns=feature_names)
        
        encoding_time = time.time()
        print(f"Input data encoded in {encoding_time - load_time:.2f} seconds")

        # Make prediction
        print("Making prediction...")
        prediction = model.predict_proba(encoded_input_df)
        
        prediction_time = time.time()
        print(f"Prediction made in {prediction_time - encoding_time:.2f} seconds")
        print(f"Total time taken: {prediction_time - start_time:.2f} seconds")

        return prediction

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None


# Example usage
team1 = 'Sri Lanka'
team2 = 'India'
prediction = predict_win_probability('male', team1, team2, 'international', 'T20', 'Colombo', team1, 'field', '')
if prediction is not None:
    print(f'Probability of {team1} winning: {prediction[0][0]}')
    print(f'Probability of {team2} winning: {prediction[0][1]}')
else:
    print("Prediction failed.")