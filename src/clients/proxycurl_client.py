from dotenv import load_dotenv
import os
import requests

load_dotenv()


class ProxycurlClient:
    """Client for interacting with the Proxycurl API."""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("PROXYCURL_API_KEY")
        if not self.api_key:
            raise ValueError("proxycurl api key unset")
        self.base_url = "https://nubela.co/proxycurl/api/v2/"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def person_search(self, params, use_cache="if-present", N=10, ids=None):
        """Search for a person using the provided parameters."""
        params["page_size"] = N
        params["use_cache"] = use_cache
        if ids:
            params["public_identifier_not_in_list"] = ids
        
  
        try:
            response = requests.get(
                f"{self.base_url}search/person",
                params=params,
                headers=self.headers,
            )
            
            if response.status_code == 200 and response.content:
                return response.json()
            else:
                print(f"Error in person_search: Status {response.status_code}")
                return {
                    'error': f'API Error: {response.status_code}'
                }
        except Exception as e:
            print(f"Exception in person_search: {e}")
            return None
    
    def fetch_linkedin_profile(self, url, use_cache="if-present"):
        """Fetch LinkedIn profile data for the given URL."""
        params = {
            'linkedin_profile_url': url,
            'use_cache': use_cache,
        }
        
        try:
            response = requests.get(f'{self.base_url}linkedin', params=params, headers=self.headers)
            
            if response.status_code == 200 and response.content:
                profile = response.json()
                profile['input_linkedin_url'] = url
                return profile
            else:
                print(f"Error fetching profile {url}: Status {response.status_code}")
                return {
                    'input_linkedin_url': url,
                    'public_identifier': url.split('/')[-1].split('?')[0],
                    'error': f'API Error: {response.status_code}'
                }
        except Exception as e:
            print(f"Exception fetching profile {url}: {e}")
            return None

