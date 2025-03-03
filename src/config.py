import numpy as np

class Config:
    MATRIX = {
        # Undergrad scoring
        'UNDERGRAD': {
            'TIERS': {
                1: set(),  # Tier 1 (Other)
                2: {  # Tier 2
                    "Carnegie Mellon University", "Brigham Young University", "Georgia Institute of Technology", 
                    "Purdue", "Tsinghua University", "IIT Delhi", "Hebrew University", "University of Virginia", 
                    "University of Washington", "Dartmouth", "Penn State University", "UC San Diego", 
                    "University of Maryland", "University of Colorado", "Georgetown", "Reichman", 
                    "Univeristy of British Columbia", "NSU", "IIT Kharagpur", "University of Florida", 
                    "Ben Gurion University", "University of North Carolina", "Indiana University", "University of Oxford"
                },
                3: {  # Tier 3
                    "Stanford", "Massachusetts Institute of Technology", "University of California, Berkeley", 
                    "University of Pennsylvania", "Harvard", "Cornell", "Tel Aviv University", "UMich", 
                    "University of Texas", "University of Illinois", "Columbia", "Yale", "UCLA", "Princeton", 
                    "USC", "Technion - Israel Institute of Technology", "Duke", "Northwestern", "IIT Bombay", 
                    "NYU", "University of Waterloo", "Brown", "McGill", "University of Wisconsin", "University of Toronto"
                }
            },
            'WEIGHT': 2,
            'DIMENSION': 3
        },
        
        # Graduate school scoring
        'GRADUATE': {
            'TIERS': {
                0: {'None'},  # Tier 0 - No graduate degree
                1: set(),  # Tier 1 (other) 
                2: set(),  # Tier 2 
                3: {  # Tier 3
                    "Stanford", "Massachusetts Institute of Technology", "University of California, Berkeley", 
                    "University of Pennsylvania", "Harvard", "Columbia", "Northwestern", 
                    "New York University", "University of Cambridge", "University of Oxford"
                }
            },
            'WEIGHT': 2,
            'DIMENSION': 4
        },
        
        # Company quality scoring
        'COMPANY': {
            'TIERS': {
                1: set(),  # Tier 1 (other)
                2: {"Google", "Tesla"},  # Tier 2
                3: {"Anduril", "Palantir", "Ramp", "OpenAI", "Anthropic"}  # Tier 3
            },
            'WEIGHT': 4,
            'DIMENSION': 3
        },
        
        # Seniority scoring
        'SENIORITY': {
            'TIERS': {
                1: set(),  # Tier 1 (other)
                2: {'Head', 'Senior', 'Vice President'},  # Tier 2
                3: {'Executive', 'SVP', 'VP', "Senior Vice President", 'Director', 'Principal'}  # Tier 3
            },
            'WEIGHT': 3,
            'DIMENSION': 3
        },
        
        # Expertise scoring
        'EXPERTISE': {
            'TIERS': {
                1: set(),  # Tier 1 (other)
                2: set(),  # Tier 2
                3: {'Engineering', 'Product', 'Engineer'}  # Tier 3
            },
            'WEIGHT': 3,
            'DIMENSION': 3
        },
        
        # Previous exit scoring
        'EXIT': {
            'TIERS': {
                0: 'No',  # Tier 0
                1: '$1-25M',  # Tier 1
                2: '$25-100M',  # Tier 2
                3: '$100M+'  # Tier 3
            },
            'WEIGHT': 5,
            'DIMENSION': 4
        },
        
        # Previous founder experience scoring
        'FOUNDER': {
            'TIERS': {
                1: 'No',  # Tier 1
                2: 'Yes - unsuccessful',  # Tier 2
                3: 'Yes - successful'  # Tier 3
            },
            'WEIGHT': 1,
            'DIMENSION': 3
        },
        
        # Startup experience scoring
        'STARTUP': {
            'TIERS': {
                1: 'No startup experience',  # Tier 1
                2: 'Early at startup',  # Tier 2
                3: 'Early at successful startup'  # Tier 3
            },
            'WEIGHT': 1,
            'DIMENSION': 3
        }
    }

    SYNTH = {
        'POPULATIONS' : {
            "successful": {
                "fraction": 0.25,
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.10, 0.40, 0.50]),
                    "GRADUATE":  np.array([0.30, 0.20, 0.20, 0.30]),
                    "EXIT":      np.array([0.60, 0.15, 0.10, 0.15]),
                    "FOUNDER":   np.array([0.40, 0.30, 0.30]),
                    "STARTUP":   np.array([0.30, 0.30, 0.40]),
                    "COMPANY":   np.array([0.20, 0.30, 0.50]),
                    "SENIORITY": np.array([0.20, 0.30, 0.50]),
                    "EXPERTISE": np.array([0.20, 0.30, 0.50])
                },
                "p_funding": 0.80,
                "mu_funding": 15.0,
                "sigma_funding": 1.2,
                "p_exit": 0.30,
                "mu_exit": 18.0,
                "sigma_exit": 1.3
            },
            "midtier": {
                "fraction": 0.25,
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.30, 0.50, 0.20]),
                    "GRADUATE":  np.array([0.50, 0.20, 0.20, 0.10]),
                    "EXIT":      np.array([0.85, 0.10, 0.03, 0.02]),
                    "FOUNDER":   np.array([0.55, 0.30, 0.15]),
                    "STARTUP":   np.array([0.50, 0.30, 0.20]),
                    "COMPANY":   np.array([0.50, 0.30, 0.20]),
                    "SENIORITY": np.array([0.40, 0.35, 0.25]),
                    "EXPERTISE": np.array([0.40, 0.30, 0.30])
                },
                "p_funding": 0.50,
                "mu_funding": 14.5,
                "sigma_funding": 1.0,
                "p_exit": 0.10,
                "mu_exit": 17.0,
                "sigma_exit": 1.2
            },
            "control": {
                "fraction": 0.50,
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.80, 0.15, 0.05]),
                    "GRADUATE":  np.array([0.75, 0.15, 0.05, 0.05]),
                    "EXIT":      np.array([0.95, 0.03, 0.01, 0.01]),
                    "FOUNDER":   np.array([0.70, 0.20, 0.10]),
                    "STARTUP":   np.array([0.65, 0.25, 0.10]),
                    "COMPANY":   np.array([0.70, 0.20, 0.10]),
                    "SENIORITY": np.array([0.70, 0.20, 0.10]),
                    "EXPERTISE": np.array([0.60, 0.20, 0.20])
                },
                "p_funding": 0.15,
                "mu_funding": 14.0,
                "sigma_funding": 0.8,
                "p_exit": 0.02,
                "mu_exit": 16.0,
                "sigma_exit": 1.2
            }
        },
        'alpha' : 2.0,
        'beta'  : 10.0,
        'noise_std' : 0.1
    }

cfg = Config()