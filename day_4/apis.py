import requests

def fetch_posts():
    try:
        response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
        response.raise_for_status()  # Raises an HTTPError for bad responses
        if response.status_code == 200:
            data = response.json()
            print("Data retrieved successfully:")
            return data
        else:
            print("Failed to retrieve data:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("An error occurred:", e)