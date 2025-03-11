import re
import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from difflib import get_close_matches
from src.config.config import cfg
from src.data.profile_transforms import ProfileTransforms
from src.utils.profile_utils import save_profiles

def process_batch_data(batch_code, synth_data_path=None, profiles_path=None, funding_path=None, output_path=None):
    """
    Process batch data by matching profiles with funding information and computing success outcomes.
    """
    if synth_data_path is None:
        synth_data_path = '../data/synth/encoded_founders_composites.csv'
    if profiles_path is None:
        profiles_path = f'../data/raw/{batch_code}_profiles.csv'
    if funding_path is None:
        funding_path = f'../data/live/yc/{batch_code}.csv'
    if output_path is None:
        output_path = f'../data/encoded/{batch_code}_encoded_with_outcomes.csv'
    
    save_df = pd.read_csv(synth_data_path)
    profiles_df = pd.read_csv(profiles_path)
    funding_df = pd.read_csv(funding_path)
    
    result_df = pd.DataFrame(columns=save_df.columns)
    
    SUCCESS_FUNDING_THRESHOLD = cfg.SUCCESS_FUNDING_THRESHOLD
    
    funding_df['normalized_name'] = funding_df['Name'].apply(normalize_name)
    
    normalized_funding_names = list(funding_df['normalized_name'])
    name_to_idx = {name: idx for idx, name in enumerate(normalized_funding_names) if name}
    
    match_log = []
    
    for _, row in profiles_df.iterrows():
        feature_str = row['feature_vector']
        name = row['Name']
        normalized_name = normalize_name(name)
        
        feature_str = feature_str.replace('[', '').replace(']', '').replace(',', ' ')
        feature_values = [float(x) for x in feature_str.split() if x.strip()]
        
        new_row = {}
        
        for i, col in enumerate(save_df.columns[:26]):
            if i < len(feature_values):
                new_row[col] = feature_values[i]
        
        funding_data, match_type, matched_name = find_matching_funding_data(
            normalized_name, funding_df, normalized_funding_names, name_to_idx
        )
        
        match_log.append({
            'profile_name': name,
            'matched_funding_name': matched_name,
            'match_type': match_type
        })
        
        new_row['exit_value'] = funding_data.iloc[0]['exit_value_usd']
        new_row['funding_amount'] = funding_data.iloc[0]['total_funding_usd']
        
        exit_value = funding_data.iloc[0]['exit_value_usd']
        funding_amount = funding_data.iloc[0]['total_funding_usd']
        new_row['success'] = 1 if (exit_value > 0 or funding_amount > SUCCESS_FUNDING_THRESHOLD) else 0
        
        result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
    
    print_summary_statistics(result_df, match_log)
    
    result_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    
    match_log_df = pd.DataFrame(match_log)
    
    return result_df, match_log_df

def process_batch_file(file_path, linkedin_profiles, processed_urls, proxycurl_client, 
                      output_path=None, batch_size=5, sleep_time=1):
    """Process a batch file of YC founders, fetching their LinkedIn profiles."""
    print(f"Processing {file_path}...")
    
    founders = pd.read_csv(file_path)
    founders = founders.dropna(subset=['LinkedIn'])
    batch_code = os.path.basename(file_path).split('.')[0]
    
    new_profiles = []
    
    for idx, row in founders.iterrows():
        linkedin_url = row['LinkedIn']
        if linkedin_url in processed_urls:
            print(f"Skipping already processed profile: {linkedin_url}")
            continue
            
        print(f"Fetching profile {idx+1}/{len(founders)}: {linkedin_url}")
        profile = proxycurl_client.fetch_linkedin_profile(linkedin_url)
        
        if profile:
            profile['yc_batch'] = batch_code
            profile['company_name'] = row.get('Company', '')
            linkedin_profiles.append(profile)
            new_profiles.append(profile)
            processed_urls.add(linkedin_url)
            
            if len(new_profiles) % batch_size == 0:
                save_profiles(linkedin_profiles, output_path)
        
        time.sleep(sleep_time)
    
    if new_profiles:
        save_profiles(linkedin_profiles, output_path)
        
    return linkedin_profiles, processed_urls

def evaluate_batch_companies(batch_codes, output_dir, perplexity_client):
    """
    Evaluate companies in YC batches to get their exit values and funding amounts.
    """
    for bc in batch_codes:
        try:
            filename = f'{output_dir}/{bc}.csv'
            
            batch_df = pd.read_csv(filename)
            batch_df = batch_df.drop_duplicates(subset=['Name', 'LinkedIn'])
            batch_df = batch_df.dropna(subset=['Name', 'LinkedIn'])
            
            for company in batch_df['Company'].unique():
                try:
                    evaluation = perplexity_client.eval_company(f'{company} (YC {bc})')
                    
                    batch_df.loc[batch_df['Company'] == company, 'exit_value_usd'] = evaluation.get('exit_value_usd')
                    batch_df.loc[batch_df['Company'] == company, 'total_funding_usd'] = evaluation.get('total_funding_usd')
                    
                    batch_df.to_csv(filename, index=False)
                    print(f"Processed {company}")
                    
                except Exception as e:
                    print(f"Error evaluating {company} (YC {bc}): {e}")
            
        except Exception as e:
            print(f"Error loading {bc}: {e}")


def normalize_name(name):
    """
    Normalize a name for comparison by removing special characters and converting to lowercase.
    """
    if not isinstance(name, str):
        return ""
    name = re.sub(r'[^\w\s]', '', name)
    return name.lower().strip()

def find_matching_funding_data(normalized_name, funding_df, normalized_funding_names, name_to_idx):
    """
    Find matching funding data for a profile using various matching strategies.
    """
    funding_data = funding_df[funding_df['normalized_name'] == normalized_name]
    
    if not funding_data.empty:
        match_type = "exact"
        matched_name = funding_data.iloc[0]['Name']
    else:
        if normalized_name:
            close_matches = get_close_matches(normalized_name, normalized_funding_names, n=3, cutoff=0.6)
            
            if close_matches:
                best_match = close_matches[0]
                match_idx = name_to_idx[best_match]
                funding_data = funding_df.iloc[[match_idx]]
                match_type = "fuzzy"
                matched_name = funding_data.iloc[0]['Name']
            else:
                first_word = normalized_name.split()[0] if normalized_name.split() else ""
                if first_word:
                    first_word_matches = [name for name in normalized_funding_names if name.startswith(first_word)]
                    if first_word_matches:
                        best_match = first_word_matches[0]
                        match_idx = name_to_idx[best_match]
                        funding_data = funding_df.iloc[[match_idx]]
                        match_type = "first_word"
                        matched_name = funding_data.iloc[0]['Name']
                    else:
                        funding_data = funding_df.iloc[[0]]
                        match_type = "default"
                        matched_name = funding_data.iloc[0]['Name']
                else:
                    funding_data = funding_df.iloc[[0]]
                    match_type = "default"
                    matched_name = funding_data.iloc[0]['Name']
        else:
            funding_data = funding_df.iloc[[0]]
            match_type = "default"
            matched_name = funding_data.iloc[0]['Name']
    
    return funding_data, match_type, matched_name

def print_summary_statistics(result_df, match_log):
    """
    Print summary statistics about processed data and match quality.
    """
    print(f"Processed {len(result_df)} profiles")
    print(f"Found funding data for {result_df['funding_amount'].notna().sum()} profiles")
    print(f"Found exit data for {result_df['exit_value'].notna().sum()} profiles")
    print(f"Successful companies: {result_df['success'].sum()}")

    print("\nMatch Verification (first 20 entries):")
    for i, match in enumerate(match_log[:20]):
        print(f"{i+1}. {match['profile_name']} → {match['matched_funding_name']} ({match['match_type']})")

    match_types = pd.Series([m['match_type'] for m in match_log]).value_counts()
    print("\nMatch type distribution:")
    for match_type, count in match_types.items():
        print(f"{match_type}: {count} ({count/len(match_log)*100:.1f}%)")

def process_top_companies(top_companies_path, linkedin_profiles_path, perplexity_client, 
                          transforms=None, output_path=None, sleep_time=0.5):
    """
    Special handling of the top_companies flag
    """
    if output_path is None:
        output_path = '../data/raw/top_companies_profiles.csv'
    
    with open(linkedin_profiles_path, 'r') as f:
        linkedin_profiles = json.load(f)
    print(f"Loaded {len(linkedin_profiles)} LinkedIn profiles")
    
    top_companies = pd.read_csv(top_companies_path)
    print(f"Processing {len(top_companies)} top companies")
    
    if transforms is None:
        transforms = ProfileTransforms({}, cfg.MATRIX)
    
    processed_profiles = []
    
    for idx, company in top_companies.iterrows():
        print(f"\nProcessing {idx+1}/{len(top_companies)}: {company['Company']}")
        
        profile = None
        linkedin_url = company['LinkedIn'].lower().strip().rstrip('/')
        
        for p in linkedin_profiles:
            if any(url and linkedin_url in str(url).lower().strip().rstrip('/') 
                   for url in [p.get('linkedin_profile_url'), p.get('public_identifier'), 
                             p.get('input_linkedin_url'), p.get('public_url')]):
                profile = p
                break
        
        if not profile:
            print(f"No profile found for {linkedin_url}")
            continue
    
        cutoff_date = company['cutoff_date'] if pd.notna(company['cutoff_date']) else None
        
        try:
            processed = transforms.process_profile(profile, cutoff_date)
            if not processed:
                print("Failed to process profile")
                continue
            
            try:
                evaluation = perplexity_client.eval_person(processed, cfg.MATRIX)
                time.sleep(sleep_time)  
            except Exception as e:
                print(f"Error in AI evaluation: {e}")
                evaluation = {"exited_founder": 0, "previous_founder": 1, "startup_experience": 1}
            
            processed.update({
                'Company': company['Company'],
                'exit_value_usd': company['exit_value_usd'],
                'total_funding_usd': company['total_funding_usd'],
                'EXIT': 1 if company['exit_value_usd'] > 0 else evaluation.get('exited_founder', 0),
                'FOUNDER': evaluation.get('previous_founder', 1),
                'STARTUP': evaluation.get('startup_experience', 1)
            })
            
            temp_df = pd.DataFrame([processed])
            transforms.df = temp_df
            transforms._add_ordinal_columns()
            feature_matrix = transforms.create_feature_matrix()
            processed['feature_vector'] = feature_matrix[0].tolist()
            
            processed_profiles.append(processed)
            print(f"Successfully processed {processed['Name']}")
            
        except Exception as e:
            print(f"Error processing profile: {str(e)}")
            continue
    
    results_df = pd.DataFrame(processed_profiles)
    
    if not results_df.empty:
        results_df.to_csv(output_path, index=False)
        print(f"\nProcessed {len(results_df)} profiles successfully")
        print(f"Saved to {output_path}")
    else:
        print("No profiles were successfully processed")
    
    return results_df
