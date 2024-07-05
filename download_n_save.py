from zipfile import ZipFile
import re
import mysql.connector
import os
import requests
import time
import json

process_start_time = time.time()

# go to the url and download the zip folder in current directory
def download_zip_file(url):
    # Extract the file name from the URL
    file_name = url.split('/')[-1]

    # Determine the destination directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Send a GET request to the URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Define the file path
        file_path = os.path.join(script_dir, file_name)

        start_time = time.time()
        downloaded_bytes = 0

        # Save the file to the destination directory
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded_bytes += len(chunk)

                    elapsed_time = int(time.time() - start_time)
                    print(
                        f"Download started, Running since {elapsed_time} seconds elapsed.", end='\r')

        end_time = time.time()
        duration = end_time - start_time

        print(
            f"File '{file_name}' downloaded and saved successfully in {duration} seconds.")
    else:
        print("Error: Failed to download the file.")


# Example usage
url = "https://cricsheet.org/downloads/all_json.zip"
download_zip_file(url)

# do a basic check of database and table existence


def check_database(database_name):
    print("Database check started")

    start_time = time.time()

    # Connect to the MySQL server
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456"
        )

        cursor = connection.cursor()

        # Check if the database exists
        cursor.execute("SHOW DATABASES")
        databases = cursor.fetchall()
        database_exists = False

        for db in databases:
            if db[0] == database_name:
                database_exists = True
                break

        # If the database doesn't exist, create it
        if not database_exists:
            cursor.execute(f"CREATE DATABASE {database_name}")
            print(f"Database '{database_name}' created")
        else:
            print(f"Database '{database_name}' exists")

        # Use the database
        cursor.execute(f"USE {database_name}")

        # List of required tables
        required_tables = [
            "matches",
            "meta_info",
            "info_section",
            "innings_info",
            "over_section"
        ]

        # Check and create missing tables
        for table in required_tables:
            cursor.execute(f"SHOW TABLES LIKE '{table}'")
            table_exists = cursor.fetchone()

            if table_exists:
                print(f"Table '{table}' exists")
            else:
                # Create the table
                create_table_query = None
                if table == "matches":
                    create_table_query = """
                        CREATE TABLE matches (
                            start_date DATE,
                            teams_type VARCHAR(255),
                            match_type VARCHAR(255),
                            gender VARCHAR(255),
                            match_id VARCHAR(50) PRIMARY KEY,
                            team_involved_one VARCHAR(255),
                            team_involved_two VARCHAR(255)
                        )
                    """
                elif table == "meta_info":
                    create_table_query = """
                        CREATE TABLE meta_info (
                            match_id VARCHAR(255) PRIMARY KEY,
                            data_version VARCHAR(255),
                            created VARCHAR(255),
                            revision VARCHAR(255)
                        )
                    """
                elif table == "info_section":
                    create_table_query = """
                        CREATE TABLE info_section (
                            match_id VARCHAR(255) PRIMARY KEY,
                            balls_per_over INT,
                            bowl_out TEXT,
                            city VARCHAR(255),
                            dates TEXT,
                            event TEXT,
                            gender VARCHAR(255),
                            match_type VARCHAR(255),
                            match_type_number INT,
                            missing VARCHAR(255),
                            officials TEXT,
                            outcome TEXT,
                            overs INT,
                            player_of_match TEXT,
                            players TEXT,
                            registry TEXT,
                            season TEXT,
                            supersubs VARCHAR(255),
                            team_type VARCHAR(255),
                            teams TEXT,
                            toss TEXT,
                            venue VARCHAR(255)
                        )
                    """
                elif table == "innings_info":
                    create_table_query = """
                        CREATE TABLE innings_info (
                            match_id VARCHAR(255),
                            team VARCHAR(255),
                            declared VARCHAR(255),
                            forfeited VARCHAR(255),
                            powerplays TEXT,
                            target TEXT,
                            super_over TEXT,
                            innings INT
                        )
                    """
                elif table == "over_section":
                    create_table_query = """
                        CREATE TABLE over_section (
                            match_id VARCHAR(255),
                            innings VARCHAR(255),
                            overs VARCHAR(10),
                            batter VARCHAR(255),
                            bowler VARCHAR(255),
                            extras JSON,
                            non_striker VARCHAR(255),
                            review JSON,
                            runs JSON,
                            wickets JSON,
                            replacements JSON
                        )
                    """

                if create_table_query:
                    cursor.execute(create_table_query)
                    print(f"Table '{table}' created")

        connection.commit()
        cursor.close()
        connection.close()

        end_time = time.time()
        duration = end_time - start_time

        print(f"Database check completed in {duration} seconds")

    except mysql.connector.Error as error:
        print("Error while connecting to MySQL:", error)


# Example usage
database_name = "cricket_data"
check_database(database_name)

time.sleep(5)

def extract_patterns_from_text(text):
    pattern = r'(\d{4}-\d{2}-\d{2}) - (club|international) - (Test|ODI|ODM|T20|IT20|MDM|\w+) - (male|female) - ([\w\s]+) - (.+)'
    matches = re.findall(pattern, text)
    processed_matches = []
    for match in matches:
        match_list = list(match)
        teams = match_list.pop()  # Extract the teams string from the match tuple
        team1, team2 = teams.split(' vs ')  # Split teams at 'vs' separator
        match_list.append(team1.strip())  # Append first team to the match list
        match_list.append(team2.strip())  # Append second team to the match list
        processed_matches.append(tuple(match_list))  # Append processed match tuple to the list
    return processed_matches


def get_existing_match_ids():
    try:
        # Connect to the MySQL server
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="cricket_data"  # Update with your database name
        )

        cursor = connection.cursor()

        # Execute SQL query to select existing match IDs from the database
        cursor.execute("SELECT match_id FROM matches")

        # Fetch all rows
        existing_match_ids = cursor.fetchall()

        # Convert the result to a list of match IDs
        existing_match_ids = [row[0] for row in existing_match_ids]

        cursor.close()
        connection.close()

        return existing_match_ids

    except mysql.connector.Error as error:
        print("Error while connecting to MySQL:", error)
        return []

def data_processing():
    print("Starting data check and storage")

    # Specify the name of the zip file and the file within it
    zip_file_name = 'all_json.zip'  # Replace with the actual zip file name
    file_within_zip = 'README.txt'  # Replace with the actual file name within the zip

    # Extract the patterns and Match IDs from the file within the zip
    with ZipFile(zip_file_name, 'r') as zip:
        with zip.open(file_within_zip) as file:
            data = file.read().decode('utf-8')
            patterns = extract_patterns_from_text(data)
            readme_match_ids = set([pattern[4] for pattern in patterns])
            print("readme match ids:",readme_match_ids)
    
            # Get existing match IDs from the database
            existing_match_ids = set(get_existing_match_ids())
            

            #compare the README.txt match_id vs database existing match_id and produce the match_id which are present in the README.txt but not present in the database
            missing_match_ids = readme_match_ids - existing_match_ids
            print("missing match_id:", missing_match_ids)
            
            # Write missing match ID data into the database
            for match_id in missing_match_ids:
                # Find the pattern corresponding to the current match ID
                for pattern in patterns:
                    if pattern[4] == match_id:
                        # Insert data into the matches table
                        insert_into_matches(match_id, pattern)
                        
                        #collect meta info
                        collect_meta(match_id)

                        #collect info section
                        collect_info(match_id)

                        #collect over section
                        collect_over_section(match_id)

                        #collect innings information
                        collect_innings_info(match_id)
            
            

# Define a function to insert data into the matches table
def insert_into_matches(match_id, pattern):
    try:
        # Connect to the MySQL server
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="cricket_data"  # Update with your database name
        )

        cursor = connection.cursor()

        # Define the SQL query to insert data into the matches table
        insert_query = """
            INSERT INTO matches (start_date, teams_type, match_type, gender, match_id, team_involved_one, team_involved_two)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        print("Pattern:", pattern)

        # Execute the SQL query with the match information
        cursor.execute(insert_query, pattern)

        connection.commit()
        cursor.close()
        connection.close()

        # print(f"Data for match ID '{match_id}' inserted into the matches table")

    except mysql.connector.Error as error:
        print("Error while connecting to MySQL:", error)

def collect_meta(match_id):
    print(f"Reading {match_id}.json for meta information")

    start_time = time.time()

    try:
        # Open the zip file
        with ZipFile("all_json.zip", 'r') as zip_file:
            # Extract the match_id.json file from the zip archive
            with zip_file.open(f"{match_id}.json") as file:
                # Read the JSON data from the file
                data = json.load(file)
                meta = data["meta"]
                data_version = meta["data_version"]
                created = meta["created"]
                revision = meta["revision"]

                # Connect to the MySQL server
                connection = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="123456",
                    database="cricket_data"  # Update with your database name
                )

                cursor = connection.cursor()

                # Define the insert query for the meta_info table
                insert_query_meta = """
                    INSERT INTO meta_info (match_id, data_version, created, revision)
                    VALUES (%s, %s, %s, %s)
                """

                # Execute the insert query for the meta section
                cursor.execute(insert_query_meta, (match_id, data_version, created, revision))
                print("match_id:",match_id, type(match_id))
                print("data_version:",data_version, type(data_version))
                print("created:",created, type(created))
                print("revision:", revision, type(revision))

                connection.commit()
                cursor.close()
                connection.close()

                print(f"Meta information for match ID '{match_id}' inserted into the meta_info table")

    except KeyError as key_error:
        print(f"Error: '{key_error}' key not found in {match_id}.json file.")
    except FileNotFoundError:
        print(f"Error: {match_id}.json file not found in the zip archive.")

    end_time = time.time()
    duration = end_time - start_time

    print(f"Collected and inserted meta information from {match_id}.json in {duration} seconds")


def collect_info(match_id):
    start_time = time.time()
    print(f"Reading {match_id}.json for info section")
    try:
        with ZipFile("all_json.zip", 'r') as zip_file:
            with zip_file.open(f"{match_id}.json") as file:
                data = json.load(file)

                info = data.get("info")
                if info:
                    balls_per_over = json.dumps(info.get("balls_per_over"))
                    bowl_out = json.dumps(info.get("bowl_out"))
                    city = json.dumps(info.get("city"))
                    # event = json.dumps(info.get("event"))
                    gender = json.dumps(info.get("gender"))
                    match_type = json.dumps(info.get("match_type"))
                    match_type_number = info.get("match_type_number")
                    # missing = info.get("missing")
                    # officials = json.dumps(info.get("officials"))
                    # outcome = json.dumps(info.get("outcome"))
                    overs = info.get("overs")
                    # player_of_match = json.dumps(info.get("player_of_match"))
                    # players = json.dumps(info.get("players"))
                    # registry = json.dumps(info.get("registry"))
                    season = json.dumps(info.get("season"))
                    supersubs = json.dumps(info.get("supersubs"))
                    team_type = json.dumps(info.get("team_type"))
                    # teams = json.dumps(info.get("teams"))
                    # toss = json.dumps(info.get("toss"))
                    venue = json.dumps(info.get("venue"))


                    print(f"DB write started for {match_id}")

                    # Connect to the MySQL database
                    connection = mysql.connector.connect(
                        host="localhost",
                        user="root",
                        password="123456",
                        database="cricket_data"
                    )

                    # Create a cursor to execute SQL queries
                    cursor = connection.cursor()

                    
                    # Convert required fields to appropriate types
                    dates_json = json.dumps(info.get("dates"))
                    event_json = json.dumps(info.get("event"))
                    missing_json = json.dumps(info.get("missing"))
                    officials_json = json.dumps(info.get("officials"))
                    outcome_json = json.dumps(info.get("outcome"))
                    player_of_match_json = json.dumps(info.get("player_of_match"))
                    players_json = json.dumps(info.get("players"))
                    registry_json = json.dumps(info.get("registry"))
                    teams_json = json.dumps(info.get("teams"))
                    toss_json = json.dumps(info.get("toss"))

                    # Define the INSERT query
                    insert_query = """
                    INSERT INTO info_section (match_id, balls_per_over, bowl_out, city, dates, event, gender, match_type,
                    match_type_number, missing, officials, outcome, overs, player_of_match, players, registry, season,
                    supersubs, team_type, teams, toss, venue)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """

                    # Execute the INSERT query with the data
                    cursor.execute(insert_query, (
                    match_id, balls_per_over, bowl_out, city, dates_json, event_json, gender, match_type, match_type_number,
                    missing_json, officials_json, outcome_json, overs, player_of_match_json, players_json, registry_json,
                    season, supersubs, team_type, teams_json, toss_json, venue
                    ))

                    print("match_id:",match_id, type(match_id))
                    print("balls_per_over:",balls_per_over, type(balls_per_over))
                    print("bowl_out:",bowl_out, type(bowl_out))
                    print("city:",city, type(city))
                    print("dates_json:",dates_json, type(dates_json))
                    print("event_json:",event_json, type(event_json))
                    print("gender:",gender, type(gender))
                    print("match_type:",match_type, type(match_type))
                    print("match_type_number:",match_type_number, type(match_type_number))
                    print("missing_json:",missing_json, type(missing_json))
                    print("officials_json:",officials_json, type(officials_json))
                    print("outcome_json:",outcome_json, type(outcome_json))
                    print("overs:",overs, type(overs))
                    print("player_of_match_json:",player_of_match_json, type(player_of_match_json))
                    print("players_json:",players_json, type(players_json))
                    print("registry_json:",registry_json, type(registry_json))
                    print("season:",season, type(season))
                    print("supersubs:",supersubs, type(supersubs))
                    print("team_type:",team_type, type(team_type))
                    print("teams_json:",teams_json, type(teams_json))
                    print("toss_json:",toss_json, type(toss_json))
                    print("venue:",venue, type(venue))


                    # Commit the changes to the database
                    connection.commit()

                    # Close the cursor and connection
                    cursor.close()
                    connection.close()

                    print(f"DB success for {match_id}")

    except KeyError as key_error:
        print(f"Error: '{key_error}' key not found in {match_id}.json file.")
    except FileNotFoundError:
        print(f"Error: {match_id}.json file not found in the zip archive.")

    end_time = time.time()
    duration = end_time - start_time

    print(f"Collected info section from {match_id}.json in {duration} seconds.")


def collect_over_section(match_id):
    print(f"Reading {match_id}.json for over section")

    start_time = time.time()

    try:
        # Open the zip file and extract the JSON data
        with ZipFile("all_json.zip", 'r') as zip_file:
            with zip_file.open(f"{match_id}.json") as file:
                data = json.load(file)

                innings = data.get("innings", [])
                innings_count = len(innings)

                # Establish connection to the MySQL database
                cnx = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="123456",
                    database="cricket_data"
                )

                cursor = cnx.cursor()

                for inning_index, inning in enumerate(innings, 1):
                    overs = inning.get("overs", [])

                    over_number = 0  # Initialize the over number
                    decimal_part = 0.1  # Initialize the decimal part

                    for over in overs:
                        deliveries = over["deliveries"]

                        for delivery in deliveries:
                            batter = delivery["batter"]
                            bowler = delivery["bowler"]
                            non_striker = delivery["non_striker"]
                            runs = delivery["runs"]

                            # Check if 'extras' key exists, otherwise assign an empty dictionary
                            extras = delivery.get("extras", {})

                            # Get wickets if present, otherwise use an empty list
                            wickets = delivery.get("wickets", [])

                            # Get review if present, otherwise use an empty dictionary
                            review = delivery.get("review", {})

                            # Check if 'replacements' key exists, otherwise assign an empty dictionary
                            replacements = delivery.get("replacements", {})

                            # Format the over display to have a single decimal place
                            over_display = f"{over_number}.{int(decimal_part * 10)}"
                            
                            # Convert dictionary values to JSON strings
                            extras = json.dumps(extras)
                            review = json.dumps(review)
                            runs = json.dumps(runs)
                            wickets = json.dumps(wickets)
                            replacements = json.dumps(replacements)
                            
                            # Insert the data into the database table
                            insert_query = """
                            INSERT INTO over_section (match_id, innings, overs, batter, bowler, extras, non_striker, review, runs, wickets, replacements)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """

                            data_values = (match_id, f"{inning_index}/{innings_count}", over_display, batter, bowler, extras, non_striker, review, runs, wickets, replacements)
                            
                            print("match_id:",match_id, type(match_id))
                            print("inning_index/innings_count:", type(f"{inning_index}/{innings_count}"))
                            print("over_display:",over_display, type(over_display))
                            print("batter:",batter, type(batter))
                            print("bowler:",bowler, type(bowler))
                            print("extras:",extras, type(extras))
                            print("non_striker:",non_striker, type(non_striker))
                            print("review:",review, type(review))
                            print("runs:",runs, type(runs))
                            print("wickets:",wickets, type(wickets))
                            print("replacements:",replacements, type(replacements))



                            cursor.execute(insert_query, data_values)

                            cnx.commit()

                            decimal_part += 0.1  # Increment the decimal part by 0.1 for each delivery

                        # Reset the decimal part to 0.1 for the next over
                        decimal_part = 0.1

                        # Increment the over number by 1 for the next over
                        over_number += 1

                cursor.close()
                cnx.close()

    except FileNotFoundError:
        print(f"Error: {match_id}.json file not found in the zip archive.")

    end_time = time.time()
    duration = end_time - start_time  
    
    print(f"DB write started for {match_id}")
    print(f"DB success for {match_id}")
    
    print(f"Collected over section from {match_id}.json in {duration} seconds")

def collect_innings_info(match_id):
    print(f"Reading {match_id}.json for innings info")
    start_time = time.time()
    try:
        # Open the zip file and extract the JSON data
        with ZipFile("all_json.zip", 'r') as zip_file:
            with zip_file.open(f"{match_id}.json") as file:
                data = json.load(file)

                innings = data.get("innings", [])
                innings_count = len(innings)

                # Establish connection to the MySQL database
                cnx = mysql.connector.connect(
                    host="localhost",
                    user="root",
                    password="123456",
                    database="cricket_data"
                )

                cursor = cnx.cursor()

                for inning_index, inning_data in enumerate(innings, 1):
                    team = inning_data.get("team", "")
                    declared = inning_data.get("declared", "")
                    forfeited = inning_data.get("forfeited", "")
                    powerplays = json.dumps(inning_data.get("powerplays", {}))
                    target = json.dumps(inning_data.get("target", {}))
                    super_over = json.dumps(inning_data.get("super_over", {}))

                    # Insert the data into the database table
                    insert_query = """
                    INSERT INTO innings_info (match_id, team, declared, forfeited, powerplays, target, super_over, innings)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """

                    data_values = (match_id, team, declared, forfeited, powerplays, target, super_over, inning_index)

                    print("match_id:",match_id, type(match_id))
                    print("team:",team, type(team))
                    print("declared:",declared, type(declared))
                    print("forfeited:",forfeited, type(forfeited))
                    print("powerplays:",powerplays, type(powerplays))
                    print("target:",target, type(target))
                    print("super_over:",super_over, type(super_over))
                    print("inning_index:",inning_index, type(inning_index))


                    cursor.execute(insert_query, data_values)

                    cnx.commit()

                cursor.close()
                cnx.close()

    except FileNotFoundError:
        print(f"Error: {match_id}.json file not found in the zip archive.")

    end_time = time.time()
    duration = end_time - start_time  
    
    print(f"DB write started for {match_id}")
    print(f"DB success for {match_id}")
    
    print(f"Collected innings info section from {match_id}.json in {duration} seconds")


# Example usage
start_data_processing = data_processing()

process_end_time = int(time.time() - process_start_time)
print(f"Total time taken to complete the process {process_end_time} secs")