import pandas as pd

#load the dataset
file_path = 'matches_data.csv'
df = pd.read_csv(file_path)

# Drop the specified columns
df = df.drop(columns=['runs', 'innings', 'wickets', 'method'])

# Fill missing values in the 'winner' column with corresponding values from the 'result' column
df['winner'] = df['winner'].fillna(df['result'])

# Drop the 'result' column
df = df.drop(columns=['result'])

# Verify the changes
print("Columns after dropping and filling missing values:")
print(df.info())

print("\nCheck for missing values:")
print(df.isnull().sum())

# Display the first few rows of the dataset after processing
print("First 5 rows of the dataset after processing:")
print(df.head())

# Define the match types and corresponding filenames
match_types = ['IPL', 'ODI', 'Test', 'T20']
output_files = ['matches_ipl.csv', 'matches_odi.csv', 'matches_test.csv', 'matches_t20.csv']

# Create and save filtered datasets
for match_type, output_file in zip(match_types, output_files):
    filtered_df = df[df['match_type'] == match_type]
    filtered_df.to_csv(output_file, index=False)
    print(f"Created {output_file} with {len(filtered_df)} rows")