import json

def load_existing_profiles(path):
    try:
        with open(path, 'r') as f:
            profiles = json.load(f)
        print(f"Loaded {len(profiles)} existing profiles")
        return profiles
    except (FileNotFoundError, json.JSONDecodeError):
        print("Starting with empty profiles list")
        return []

def get_processed_urls(profiles):
    urls = set()
    for p in profiles:
        for field in ['input_linkedin_url', 'linkedin_url', 'public_url', 'url']:
            if field in p and p[field]:
                urls.add(p[field])
                break
    return urls

def get_queried_urls(data):
    profiles = [item['profile'] for item in data if 'profile' in item]
    return get_processed_urls(profiles)

def save_profiles(profiles, path):
    with open(path, 'w') as f:
        json.dump(profiles, f)