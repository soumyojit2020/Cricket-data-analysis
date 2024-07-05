from zipfile import ZipFile
import re
import mysql.connector
import os
import requests
import time
import json

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
            
            # Print only the match IDs
            for match_id in readme_match_ids:
                print(match_id)
            
            return readme_match_ids
            
            
test = data_processing()
print(test)
            
