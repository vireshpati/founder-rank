from dotenv import load_dotenv
import os
import requests

load_dotenv()


class ProxycurlClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("PROXYCURL_API_KEY")
        if not self.api_key:
            raise ValueError("proxycurl api key unset")
        self.base_url = "https://nubela.co/proxycurl/api/v2/"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    def person_search(self, params, use_cache="if-present", N=1):
        params["page_size"] = N
        params["use_cache"] = use_cache

        response = requests.get(
            f"{self.base_url}search/person",
            params=params,
            headers=self.headers,
        )
        return response.json()

    def get_linkedin_data(self, params, use_cache="if-present"):
    
        params["use_cache"] = use_cache
        response = requests.get(
            f"{self.base_url}linkedin",
            headers=self.headers,
            params=params,
        )
        return response.json()

