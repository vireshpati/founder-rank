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

    # ... existing code ...

    SYNTH = {
        'POPULATIONS' : {
            "successful": {
                "fraction": 0.6,  # Increased from 0.5
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.05, 0.35, 0.60]),  # More emphasis on tier 3
                    "GRADUATE":  np.array([0.20, 0.15, 0.25, 0.40]),  # More emphasis on tier 3
                    "EXIT":      np.array([0.50, 0.20, 0.10, 0.20]),  # More exits
                    "FOUNDER":   np.array([0.30, 0.30, 0.40]),  # More successful founders
                    "STARTUP":   np.array([0.20, 0.30, 0.50]),  # More successful startup experience
                    "COMPANY":   np.array([0.15, 0.25, 0.60]),  # More tier 3 companies
                    "SENIORITY": np.array([0.15, 0.25, 0.60]),  # More senior roles
                    "EXPERTISE": np.array([0.15, 0.25, 0.60])   # More expertise in key areas
                },
                "p_funding": 0.95,  # Higher funding probability
                "mu_funding": 15.5,  # Higher funding amount
                "sigma_funding": 1.0,  # Tighter distribution
                "p_exit": 0.6,  # Higher exit probability
                "mu_exit": 18.5,  # Higher exit value
                "sigma_exit": 1.1  # Tighter distribution
            },
            "midtier": {
                "fraction": 0.15,  # Decreased from 0.2
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.40, 0.45, 0.15]),  # Less overlap with successful
                    "GRADUATE":  np.array([0.60, 0.25, 0.10, 0.05]),  # Less overlap with successful
                    "EXIT":      np.array([0.90, 0.07, 0.02, 0.01]),  # Clearer separation
                    "FOUNDER":   np.array([0.65, 0.25, 0.10]),  # Less overlap
                    "STARTUP":   np.array([0.60, 0.30, 0.10]),  # Less overlap
                    "COMPANY":   np.array([0.60, 0.30, 0.10]),  # Less overlap
                    "SENIORITY": np.array([0.50, 0.40, 0.10]),  # Less overlap
                    "EXPERTISE": np.array([0.50, 0.35, 0.15])   # Less overlap
                },
                "p_funding": 0.40,  # Lower than successful
                "mu_funding": 14.0,  # Lower than successful
                "sigma_funding": 0.9,  # Tighter distribution
                "p_exit": 0.08,  # Lower exit probability
                "mu_exit": 16.5,  # Lower exit value
                "sigma_exit": 1.0  # Tighter distribution
            },
            "control": {
                "fraction": 0.25,  # Decreased from 0.4
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.90, 0.08, 0.02]),  # More extreme separation
                    "GRADUATE":  np.array([0.85, 0.10, 0.03, 0.02]),  # More extreme separation
                    "EXIT":      np.array([0.98, 0.01, 0.005, 0.005]),  # Almost no exits
                    "FOUNDER":   np.array([0.85, 0.10, 0.05]),  # Less founder experience
                    "STARTUP":   np.array([0.80, 0.15, 0.05]),  # Less startup experience
                    "COMPANY":   np.array([0.85, 0.10, 0.05]),  # Less prestigious companies
                    "SENIORITY": np.array([0.85, 0.10, 0.05]),  # Less senior roles
                    "EXPERTISE": np.array([0.75, 0.15, 0.10])   # Less expertise
                },
                "p_funding": 0.10,  # Lower funding probability
                "mu_funding": 13.5,  # Lower funding amount
                "sigma_funding": 0.7,  # Tighter distribution
                "p_exit": 0.01,  # Very low exit probability
                "mu_exit": 15.5,  # Lower exit value
                "sigma_exit": 1.0  # Tighter distribution
            }
        },
        'alpha' : 2.5,  # Increased from 2.0 for more separation
        'beta'  : 12.0,  # Increased from 10.0 for more separation
        'noise_std' : 0.08  # Reduced noise for clearer signal
    }

# ... existing code ...

cfg = Config()