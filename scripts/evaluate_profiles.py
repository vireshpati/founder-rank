import numpy as np
import json
import sys
import pandas as pd
import os
import argparse
from dotenv import load_dotenv

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import cfg
from src.clients.perplexity_client import PerplexityClient
from src.clients.proxycurl_client import ProxycurlClient
from src.data.profile_transforms import ProfileTransforms
from src.core.ranking import search_founders, rank_profiles, load_model
from src.utils.profile_utils import get_queried_urls

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LinkedIn profiles for founder ranking")
    parser.add_argument("--search", action="store_true", help="Search for founders instead of using existing profiles")
    parser.add_argument("--limit", type=int, default=7, help="Number of profiles to search for")
    parser.add_argument("--list-name", type=str, default="angel-network", help="Name of the list to use or create")
    parser.add_argument("--model-path", type=str, default="models/founder_rank_best.pkl", help="Path to the ranking model")
    parser.add_argument("--urls", type=str, nargs="+", help="LinkedIn URLs to evaluate")
    parser.add_argument("--save", action="store_true", help="Save the ranked results to a CSV file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    load_dotenv()
    np.set_printoptions(precision=2, suppress=True, linewidth=120)
    
    px = ProxycurlClient()
    pc = PerplexityClient()
    
    linkedin_urls = args.urls or [
        "https://linkedin.com/in/george-goodfellow/",
        "https://linkedin.com/in/adithyagurunathan/",
        "https://linkedin.com/in/sarangpujari/",
        "https://linkedin.com/in/katelam8/",
        "https://linkedin.com/in/christopher-hur/",
        "https://linkedin.com/in/aliciajsteele/",
        "https://linkedin.com/in/charlesfatunbi/",
        "https://linkedin.com/in/tejal-dahake/",
        "https://linkedin.com/in/rohan-devraj/",
        "https://linkedin.com/in/skareer/",
        "https://linkedin.com/in/imgeorgiev/",
        "https://linkedin.com/in/viresh-pati/",
    ]
    
    if args.search:
        data = search_founders(px=px, limit=args.limit)
    else:
        data = []
        try:
            with open(f"data/proxycurl/{args.list_name}.json", "r") as json_file:
                data = json.load(json_file)
            processed = get_queried_urls(data)
            
            print(f'Found {len(processed)} profiles in specified dir')
            for url in linkedin_urls:
                if url in processed:
                    print(f'already processed {url} ... skipping')
                    continue
                print(f"Fetching profile: {url}")
                profile = px.fetch_linkedin_profile(url, use_cache="if-present")
                if profile:
                    data.append({"profile": profile})
        except FileNotFoundError:
            print(f"No existing data found for {args.list_name}, fetching all profiles")
            data = []
            for url in linkedin_urls:
                print(f"Fetching profile: {url}")
                profile = px.fetch_linkedin_profile(url, use_cache="if-recent")
                if profile:
                    data.append({"profile": profile})
            
            os.makedirs("data/proxycurl", exist_ok=True)
            with open(f"data/proxycurl/{args.list_name}.json", "w") as f:
                json.dump(data, f, indent=2)
    
    T = ProfileTransforms(data)
    DF_DIR = f'data/sample_encodings/'
    os.makedirs(DF_DIR, exist_ok=True)
    
    csv_path = f'{DF_DIR}{args.list_name}-profiles.csv'
    
    if not os.path.exists(csv_path):
        print("Transforming profiles...")
        df = T.process_profiles(profiles=data, perplexity_client=pc, output_dir=DF_DIR, batch_code=args.list_name)
    else:
        print(f"Loading transformed profiles from {csv_path}")
        df = pd.read_csv(csv_path, index_col=False)
        df['feature_vector'] = df['feature_vector'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
        T.df = df
    
    print("Ranking profiles...")
    ranked_results = rank_profiles(df, T.get_feature_matrix(), model_dict=load_model(model_path=args.model_path))
    
    display_cols = ['Name', 'Linkedin', 'UNDERGRAD', 'GRADUATE', 'COMPANY', 'SENIORITY', 
                    'EXPERTISE', 'EXIT', 'FOUNDER', 'STARTUP', 'score']
    sorted_results = ranked_results[display_cols].sort_values(by="score", ascending=False)
    
    print("\nRanked Founder Profiles:")
    print("-----------------------\n")
    print(sorted_results.to_string(index=False))
    
    if args.save:
        out_dir = 'out'
        os.makedirs(out_dir, exist_ok=True)
        file_count = len([f for f in os.listdir(out_dir) if f.startswith("founders-")])
        
        output_filename = f"founders-{args.list_name}-{file_count+1}.csv"
        output_path = os.path.join(out_dir, output_filename)
        
        sorted_results.to_csv(output_path, index=False)
        print(f"\nRanked results saved to: {output_path}")

if __name__ == "__main__":
    main()