from openai import OpenAI
import os
import json
import re
from dotenv import load_dotenv
from src.utils.matrix_utils import get_tier

load_dotenv()


class PerplexityClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("perplexity api key unset")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.perplexity.ai")

    def _make_perplexity_request(self, messages, fallback, entity_name):
        try:
            response = self.client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
            )
            
  
            json_match = re.search(r"\{.*?\}", response.choices[0].message.content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                print(f"Could not extract JSON for {entity_name}. Full response: {response.choices[0].message.content}")
                return fallback

        except Exception as e:
            print(f"Error evaluating {entity_name}: {e}")
            print(f"Returning fallback data: {fallback}")
            return fallback

    def eval_person(self, person_data, matrix):
        # Profile
        name = person_data.get("Name", "Unknown")
        titles = (
            ", ".join([t for t in person_data.get("Previous Titles", []) if t])
            if person_data.get("Previous Titles")
            else "Unknown"
        )
        companies = (
            ", ".join([c for c in person_data.get("Previous Companies", []) if c])
            if person_data.get("Previous Companies")
            else "Unknown"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a venture research assistant. Your task is to evaluate a person's founder and startup experience "
                    "based on the provided information. Provide concise and structured evaluations for the categories: "
                    "'Previously an exited founder?', 'Previously a founder?', and 'Prior Startup Experience'. Use the following rating scales:\n\n"
                    f"- Previously an exited founder?\n  3: {get_tier(matrix, 'EXIT', 3)}, 2: {get_tier(matrix, 'EXIT', 2)}, 1: {get_tier(matrix, 'EXIT',1)}, 0: {get_tier(matrix, 'EXIT',0)}\n"
                    f"- Previously a founder?\n  3: {get_tier(matrix, 'FOUNDER', 3)}, 2: {get_tier(matrix, 'FOUNDER',2)}, 1: {get_tier(matrix, 'FOUNDER', 1)}'\n"
                    f"- Prior Startup Experience\n  3: {get_tier(matrix, 'STARTUP', 3)}, 2: {get_tier(matrix, 'STARTUP', 2)}, 1: {get_tier(matrix, 'STARTUP', 1)}.\n"
                    "Do not consider a a person's current experience as prior startup experience or previously a founder.\n\n"
                    "Provide your response in JSON format with keys 'exited_founder', 'previous_founder', and 'startup_experience'.\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evaluate the following person's founder and startup experience:\n\n"
                    f"Name: {name}\n"
                    f"Titles: {titles}\n"
                    f"Experiences: {companies}\n\n"
                    "Provide ratings for the categories as described."
                ),
            },
        ]

        fallback = {"exited_founder": 0, "previous_founder": 1, "startup_experience": 1}
        return self._make_perplexity_request(messages, fallback, name)

    def eval_company(self, company_name):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a venture research assistant tasked with providing information about company exits and funding.\n"
                    "Exits: Identify and report the value if the company has had an exit (IPO, acquisition, direct listing, SPAC)\n"
                    "Funding: Report the total venture funding raised in USD.\n"
                    "Return 0 for exit and funding if you cannot find any information."
                    "Output Format: Return your response in JSON format with keys exit_value_usd and total_funding_usd.\n"

                )
            },
            {
                "role": "user",
                "content": (
                    f"1. How much did {company_name} ipo|acquired|spac for?\n"
                    f"2. How much did {company_name} raise in venture capital?\n\n"
                    "Provide values for exit_value_usd and total_funding_usd as mentioned."
                )
            }
        ]
        fallback = {"exit_value_usd": 0, "total_funding_usd": 0}

        return self._make_perplexity_request(messages, fallback, company_name)
