import pandas as pd

#read the initial dataset file
matches_df = pd.read_csv('matches_data.csv')

#create a list of teams that are full time ODI members of ICC
odi_full_members = ['Australia', 'England', 'India', 'New Zealand', 'Pakistan', 'South Africa', 'Sri Lanka', 'Bangladesh', 'Zimbabwe', 'Afghanistan', 'Ireland', 'West Indies']

#filter the dataframe
filtered_df = matches_df.loc[
    (matches_df['gender'] == 'male') &
    (matches_df['match_type'] == 'ODI') &
    (matches_df['team_involved_one'].isin(odi_full_members) & matches_df['team_involved_two'].isin(odi_full_members)) &
    (~matches_df['result'].isin(['tie', 'no result', 'draw'])) &
    (~matches_df['method'].str.contains('D/L', na=False))
]

# Drop the specified columns
columns_to_drop = ['gender', 'teams_type', 'match_type', 'result', 'runs', 'innings', 'wickets', 'method']
odi_base_dataset = filtered_df.drop(columns_to_drop, axis=1)

odi_base_dataset.to_csv('odi_base_dataset.csv', index=False)
print("Generated ODI base dataset")
print("Number of records: ", len(odi_base_dataset))
print(odi_base_dataset.head(10))