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
                2: {'Head', 'Senior', 'Vice President', 'Director'},  # Tier 2
                3: {'Executive', 'SVP', 'VP', "Senior Vice President", 'Principal'}  # Tier 3
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
                "fraction": 0.5,  
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.05, 0.35, 0.60]), 
                    "GRADUATE":  np.array([0.20, 0.15, 0.25, 0.40]), 
                    "EXIT":      np.array([0.50, 0.20, 0.10, 0.20]),  
                    "FOUNDER":   np.array([0.30, 0.30, 0.40]),  
                    "STARTUP":   np.array([0.20, 0.30, 0.50]),  
                    "COMPANY":   np.array([0.15, 0.25, 0.60]), 
                    "SENIORITY": np.array([0.15, 0.25, 0.60]),  #
                    "EXPERTISE": np.array([0.15, 0.25, 0.60])   
                },
                "p_funding": 0.90,  
                "mu_funding": 15.5,  
                "sigma_funding": 1.0,  
                "p_exit": 0.6,  
                "mu_exit": 18.5,  
                "sigma_exit": 1.1  
            },
            "midtier": {
                "fraction": 0.2,  
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.40, 0.45, 0.15]), 
                    "GRADUATE":  np.array([0.60, 0.25, 0.10, 0.05]), 
                    "EXIT":      np.array([0.90, 0.07, 0.02, 0.01]),  
                    "FOUNDER":   np.array([0.65, 0.25, 0.10]), 
                    "STARTUP":   np.array([0.60, 0.30, 0.10]),  
                    "COMPANY":   np.array([0.60, 0.30, 0.10]),  
                    "SENIORITY": np.array([0.50, 0.40, 0.10]),  
                    "EXPERTISE": np.array([0.50, 0.35, 0.15])  
                },
                "p_funding": 0.40, 
                "mu_funding": 14.0,  
                "sigma_funding": 0.9,  
                "p_exit": 0.08,  
                "mu_exit": 16.5,  
                "sigma_exit": 1.0  
            },
            "control": {
                "fraction": 0.3,  #
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.90, 0.08, 0.02]),  
                    "GRADUATE":  np.array([0.85, 0.10, 0.03, 0.02]), 
                    "EXIT":      np.array([0.98, 0.01, 0.005, 0.005]),  
                    "FOUNDER":   np.array([0.85, 0.10, 0.05]), 
                    "STARTUP":   np.array([0.80, 0.15, 0.05]),  
                    "COMPANY":   np.array([0.85, 0.10, 0.05]),  
                    "SENIORITY": np.array([0.85, 0.10, 0.05]),  
                    "EXPERTISE": np.array([0.75, 0.15, 0.10])   
                },
                "p_funding": 0.10,  
                "mu_funding": 13.5,  
                "sigma_funding": 0.7,  
                "p_exit": 0.01,  
                "mu_exit": 15.5,  
                "sigma_exit": 1.0  
            }
        },
        'alpha' : 2.5, 
        'beta'  : 12.0,  
        'noise_std' : 0.08  
    }

# ... existing code ...

cfg = Config()