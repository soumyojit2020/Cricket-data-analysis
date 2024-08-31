#import required libaries
import mysql.connector
import pandas as pd
import json

#pull required data from database
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="123456",
    database="cricket_data"
)

#joining two tables we get the columns that are required
query = """SELECT matches.match_id, matches.gender, matches.start_date, matches.teams_type,
matches.match_type, matches.team_involved_one, matches.team_involved_two, info_section.outcome,
info_section.toss, info_section.city, info_section.venue
FROM matches JOIN
info_section ON
matches.match_id = info_section.match_id;"""

# Execute the query and fetch the results
cursor = connection.cursor()
cursor.execute(query)
matches_data = cursor.fetchall()

# Convert the fetched data to a pandas DataFrame
matches_df = pd.DataFrame(matches_data, columns=[desc[0] for desc in cursor.description])
print('Initial dataframe generated')
print('Sample', matches_df.head())
# Close the cursor and connection
cursor.close()
connection.close()
print('db connection closed')

#remove double quotes and unpack the toss column into winner and decision
def transform_dataframe(df):
    # 1. Change the start_date to datetime
    df['start_date'] = pd.to_datetime(df['start_date'])

    # 2. Split the toss field into toss_winner and toss_decision
    def extract_toss_details(toss):
        toss_data = json.loads(toss)  # Parse the JSON string to a dictionary
        return toss_data.get('winner'), toss_data.get('decision')
    
    df[['toss_winner', 'toss_decision']] = df['toss'].apply(lambda x: pd.Series(extract_toss_details(x)))

    # Drop the original toss column
    df.drop(columns=['toss'], inplace=True)

    # 3. Remove double quotes from the city and venue field
    df['city'] = df['city'].str.strip('"')
    df['venue'] = df['venue'].str.strip('"')

    return df

matches_df = transform_dataframe(matches_df)

def process_outcome(outcome):
    # Initialize default values
    result = winner = runs = innings = wickets = method = None

    # Parse the JSON string to a dictionary
    outcome_data = json.loads(outcome)

    # Extract the common fields
    if 'result' in outcome_data:
        result = outcome_data['result']
    if 'winner' in outcome_data:
        winner = outcome_data['winner']
    if 'method' in outcome_data:
        method = outcome_data['method']
    if 'by' in outcome_data:
        by = outcome_data['by']
        if 'runs' in by:
            runs = by['runs']
        if 'innings' in by:
            innings = by['innings']
        if 'wickets' in by:
            wickets = by['wickets']

    return result, winner, runs, innings, wickets, method

# Apply the function to the outcome column and create new columns
matches_df[['result', 'winner', 'runs', 'innings', 'wickets', 'method']] = matches_df['outcome'].apply(
    lambda x: pd.Series(process_outcome(x))
)

# Drop the original outcome column
matches_df.drop(columns=['outcome'], inplace=True)

# Define the mapping of old team names to current team names
team_name_mapping = {
    ('North-West Warriors',): 'North West Warriors',
    ('St Lucia Stars', 'St Lucia Zouks'): 'St Lucia Kings',
    ('Barbados Tridents',): 'Barbados Royals',
    ('St Lucia Zouks', 'St Lucia Kings'): 'St Lucia Kings',
	('Comilla Victorians',): 'Cumilla Warriors',
	('Chittagong Kings', 'Chittagong Vikings'): 'Chattogram Challengers',
	('Khulna Royal Bengals', 'Khulna Titans'): 'Khulna Tigers',
	('Barishal Burners', 'Barishal Bulls'): 'Fortune Barishal',
	('Dhaka Gladiators', 'Dhaka Dynamites', 'Dhaka Platoon', 'Beximco Dhaka', 'Minister Dhaka', 'Minister Group Dhaka', 'Dhaka Dominators'): 'Durdanto Dhaka',
    ('Sylhet Royals', 'Sylhet Super Stars', 'Sylhet Sixers', 'Sylhet Thunder', 'Sylhet Sunrisers'): 'Sylhet Strikers',
	('Rangpur Riders',): 'Rangpur Rangers',
	('Duronto Rajshahi', 'Rajshahi Kings'): 'Rajshahi Royals',
	('Royal Challengers Bangalore', ): 'Royal Challengers Bengaluru',
	('Delhi Daredevils',): 'Delhi Capitals',
	('Kings XI Punjab',): 'Punjab Kings',
	('Colombo Kings', 'Colombo Stars',): 'Colombo Strikers',
	('Dambulla Viiking', 'Dambulla Giants', 'Dambulla Aura',): 'Dambulla Sixers',
	('Galle Gladiators', 'Galle Titans'): 'Galle Marvels',
	('Kandy Tuskers', 'Kandy Warriors', 'Kandy Falcons'): 'B-Love Kandy',
	('Jaffna Stallions',): 'Jaffna Kings',
	# Add other mappings as needed
}

# Load the venue-city mapping from the JSON file
with open("venue_city_mapping_use.json", "r") as file:
    venue_city_mapping = json.load(file)

# Function to fill missing city values based on the venue
def fill_missing_city(row):
    if row['city'] == "null" and row['venue'] in venue_city_mapping:
        return venue_city_mapping[row['venue']]
    return row['city']

# Apply the function to fill missing city values
matches_df['city'] = matches_df.apply(fill_missing_city, axis=1)

# Check if any city values are still "null"
missing_cities = matches_df[matches_df['city'] == "null"]

# Verify the DataFrame
print(matches_df.head())

# Print the rows with "null" city values along with their venues grouped by venue
if not missing_cities.empty:
    print("Rows with 'null' city values grouped by venue:")
    grouped_missing_cities = missing_cities.groupby('venue')['city'].count()
    print(grouped_missing_cities)
else:
    print("No 'null' city values.")

# Drop the 'venue' column from the DataFrame
matches_df = matches_df.drop(columns=['venue'])

city_mapping = {
    "East London": "London",
    "Navi Mumbai": "Mumbai",
    "Kigali City": "Kigali",
    "Wong Nai Chung Gap, Hong Kong": "Wong Nai Chung Gap",
    "Dehra Dun": "Dehradun",
    "Dharamsala": "Dharmasala",
    "Delhi": "New Delhi",
    "Bangalore": "Bengaluru"
}

# Step 6: Use the mapping to replace city names in the dataframe
matches_df['city'] = matches_df['city'].replace(city_mapping)

# Verify the updated dataframe
print("\nUpdated dataframe with replaced city names:")
print(matches_df.head())

# Save the dataframe to a CSV file
matches_df.to_csv('matches_data.csv', index=False)
print('dataframe saved as matches_data.csv')