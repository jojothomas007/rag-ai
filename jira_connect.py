import requests
from dotenv import load_dotenv
import os


jira_url = os.environ["JIRA_INSTANCE_URL"]
username = os.environ["JIRA_USERNAME"]
api_token = os.environ["JIRA_API_TOKEN"]

url = f"{jira_url}/rest/api/3/project"

headers = {
    "Accept": "application/json"
}

response = requests.get(url, headers=headers, auth=(username, api_token))

if response.status_code == 200:
    projects = response.json()
    for project in projects:
        print(f"Project Key: {project['key']}, Name: {project['name']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)