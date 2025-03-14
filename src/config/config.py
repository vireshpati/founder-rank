import numpy as np


class Config:
    """Configuration class for scoring parameters."""
    
    MATRIX = {
        # Undergrad scoring
        "UNDERGRAD": {
            "TIERS": {
                1: set(),  # Tier 1 (Other)
                2: {  # Tier 2
                    "Carnegie Mellon University",
                    "Brigham Young University",
                    "Georgia Institute of Technology",
                    "Purdue University",
                    "Tsinghua University",
                    "IIT Delhi",
                    "Hebrew University",
                    "University of Virginia",
                    "University of Washington",
                    "Dartmouth College",
                    "Penn State University",
                    "UC San Diego",
                    "University of Maryland",
                    "University of Colorado",
                    "Georgetown University",
                    "Reichman",
                    "Univeristy of British Columbia",
                    "NSU",
                    "IIT Kharagpur",
                    "University of Florida",
                    "Ben Gurion University",
                    "University of North Carolina",
                    "Indiana University",
                    "University of Oxford",
                },
                3: {  # Tier 3
                    "Stanford University",
                    "Massachusetts Institute of Technology",
                    "University of California, Berkeley",
                    "University of Pennsylvania",
                    "Harvard University",
                    "Cornell University",
                    "Tel Aviv University",
                    "University of Michigan",
                    "University of Texas",
                    "University of Illinois",
                    "Columbia University",
                    "Yale University",
                    "UCLA",
                    "Princeton University",
                    "University of Southern California",
                    "Technion - Israel Institute of Technology",
                    "Duke University",
                    "Northwestern",
                    "IIT Bombay",
                    "New York University",
                    "University of Waterloo",
                    "Brown University",
                    "McGill University",
                    "University of Wisconsin",
                    "University of Toronto",
                },
            },
            "WEIGHT": 2,
            "DIMENSION": 3,
        },
        # Graduate school scoring
        "GRADUATE": {
            "TIERS": {
                0: {"None"},  # Tier 0 - No graduate degree
                1: set(),  # Tier 1 (other)
                2: set(),  # Tier 2
                3: {  # Tier 3
                    "Stanford University",
                    "Massachusetts Institute of Technology",
                    "University of California, Berkeley",
                    "University of Pennsylvania",
                    "Harvard University",
                    "Columbia University",
                    "Northwestern",
                    "New York University",
                    "University of Cambridge",
                    "University of Oxford",
                },
            },
            "WEIGHT": 2,  
            "DIMENSION": 4,
        },
        # Company quality scoring
        "COMPANY": {
            "TIERS": {
                1: set(),  # Tier 1 (other)
                2: {
                    "Google",
                    "Tesla",
                },  # Tier 2
                3: {
                    "Anduril",
                    "Palantir",
                    "Ramp",
                    "OpenAI",
                    "Anthropic",
                },  # Tier 3
            },
            "WEIGHT": 4,  
            "DIMENSION": 3,
        },
        # Seniority scoring
        "SENIORITY": {
            "TIERS": {
                1: set(),  # Tier 1 (other)
                2: {
                    "Head",
                    "Senior",
                    "Vice President",
                    "Director",
                },  # Tier 2
                3: {
                    "Executive",
                    "SVP",
                    "VP",
                    "Senior Vice President",
                    "Principal",
                },  # Tier 3
            },
            "WEIGHT": 3,  
            "DIMENSION": 3,
        },
        # Expertise scoring
        "EXPERTISE": {
            "TIERS": {
                1: set(),  # Tier 1 (other)
                2: {
                    "Research",
                    "Researcher"
                    },  # Tier 2
                3: {
                    "Engineering",
                    "Product",
                    "Engineer",
                    "Quantitative",
                },  # Tier 3
            },
            "WEIGHT": 3, 
            "DIMENSION": 3,
        },
        # Previous exit scoring
        "EXIT": {
            "TIERS": {
                0: "No",
                1: "$1-25M",
                2: "$25-100M",
                3: "$100M+",
            },  # Tier 0  # Tier 1  # Tier 2  # Tier 3
            "WEIGHT": 5,  
            "DIMENSION": 4,
        },
        # Previous founder experience scoring
        "FOUNDER": {
            "TIERS": {
                1: "No",
                2: "Yes - unsuccessful",
                3: "Yes - successful",
            },  # Tier 1  # Tier 2  # Tier 3
            "WEIGHT": 1, 
            "DIMENSION": 3,
        },
        # Startup experience scoring
        "STARTUP": {
            "TIERS": {
                1: "No startup experience",  # Tier 1
                2: "Early at startup",  # Tier 2
                3: "Early at successful startup",  # Tier 3
            },
            "WEIGHT" : 1,  
            "DIMENSION": 3,
        },
    }

    SYNTH = {
        # Tier 0/1 --> Tier 3
        "POPULATIONS": {
            "successful": {  
                "fraction": 0.40,
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.60, 0.25, 0.15]),
                    "GRADUATE": np.array([0.30, 0.45, 0.15, 0.10]),
                    "COMPANY": np.array([0.15, 0.40, 0.45]),
                    "SENIORITY": np.array([0.15, 0.35, 0.50]),
                    "EXPERTISE": np.array([0.05, 0.20, 0.75]),
                    "EXIT": np.array([0.65, 0.15, 0.10, 0.10]),
                    "FOUNDER": np.array([0.15, 0.25, 0.60]),
                    "STARTUP": np.array([0.05, 0.30, 0.65]),
                },
                "p_funding": 0.85,
                "mu_funding": 17.0,
                "sigma_funding": 1.3,
                "p_exit": 0.25,
                "mu_exit": 17.5,
                "sigma_exit": 1.5,
            },
            "midtier": {
                "fraction": 0.40,
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.70, 0.20, 0.10]),
                    "GRADUATE": np.array([0.40, 0.35, 0.20, 0.05]),
                    "COMPANY": np.array([0.30, 0.40, 0.30]),
                    "SENIORITY": np.array([0.35, 0.40, 0.25]),
                    "EXPERTISE": np.array([0.20, 0.30, 0.50]),
                    "EXIT": np.array([0.80, 0.15, 0.03, 0.02]),
                    "FOUNDER": np.array([0.40, 0.35, 0.25]),
                    "STARTUP": np.array([0.15, 0.40, 0.45]),
                },
                "p_funding": 0.65,
                "mu_funding": 16.0,
                "sigma_funding": 0.9,
                "p_exit": 0.12,
                "mu_exit": 16.5,
                "sigma_exit": 1.0,
            },
            "control": {  
                "fraction": 0.20,
                "sampling_probs": {
                    "UNDERGRAD": np.array([0.85, 0.10, 0.05]),
                    "GRADUATE": np.array([0.70, 0.20, 0.08, 0.02]),
                    "COMPANY": np.array([0.65, 0.25, 0.10]),
                    "SENIORITY": np.array([0.60, 0.30, 0.10]),
                    "EXPERTISE": np.array([0.45, 0.35, 0.20]),
                    "EXIT": np.array([0.92, 0.05, 0.02, 0.01]),
                    "FOUNDER": np.array([0.85, 0.10, 0.05]),
                    "STARTUP": np.array([0.50, 0.40, 0.10]),
                },
                "p_funding": 0.30,
                "mu_funding": 14.0,
                "sigma_funding": 0.7,
                "p_exit": 0.03,
                "mu_exit": 15.0,
                "sigma_exit": 0.9,
            }
        }
    }
    SUCCESS_FUNDING_THRESHOLD = 15000000  # ~series B
     
    FOUNDER_SEARCH_PARAMS = {
        "country": 'US',
        "education_school_name": 'Georgia Institute of Technology',  
        "current_role_title": "Founder OR Co-Founder OR 'Founding Engineer' OR CEO OR CTO OR Stealth",
        "enrich_profiles": "enrich",
        "page_size": 10,
        "use_cache": "if-present",
    }
    
    UNDERGRAD_KEYWORDS = ["bs","ba","bachelor","bachelor's", "bse","bsba","computer science", "mathematics"]
    GRAD_KEYWORDS = ['master', 'mba', 'ms', 'phd', 'jd', 'mfa', 'mfe', "master's"]
    
    THRESHOLD = 0.5

cfg = Config()