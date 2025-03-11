# Founder Rank

Founder Rank is a tool that uses AI to rank founders based on their potential to build successful startups.

## Setup

1. Create a `.env` file with required API keys:

```
PROXYCURL_API_KEY=your_proxycurl_api_key
PERPLEXITY_API_KEY=your_perplexity_api_key
```

2. Install dependencies:

```
pip install -r requirements.txt
```

## Usage

`scripts/` provides some easy usage of core functions, but all pipelines are implemented in `notebooks/` as well.

### 1. Generate Synthetic Training Data
Generate synthetic founder profiles for training:

```bash
python scripts/generate_data.py \
  --n_samples 5000 \
  --output_path data/synth/encoded_founders_composites.csv
```

### 2. Train the Model 
Train using both synthetic and real data:
```bash
# Basic training with default parameters
python scripts/train_model.py --data_path data/synth/encoded_founders_composites.csv --output_path models/founder_rank.pkl

# Train with combined synthetic and real data
python scripts/train_model.py --data_path data/synth/encoded_founders_composites.csv --real_data data/encoded/real_founders.csv --output_path models/founder_rank_combined.pkl

# Train with hyperparameter tuning enabled
python scripts/train_model.py --data_path data/synth/encoded_founders_composites.csv --tune_hyperparams --output_path models/founder_rank_optimized.pkl
```

### 3. Evaluate New Profiles
Rank new founder profiles using the trained model:
```bash
# Search for new founder profiles, save results, and use a specific model
python scripts/evaluate_profiles.py --search --limit 15 --save --model-path models/founder_rank.pkl

# Evaluate specific LinkedIn profiles with a custom list name
python scripts/evaluate_profiles.py --urls "https://linkedin.com/in/username1" "https://linkedin.com/in/username2" --list-name "potential-founders" --save

# Complete workflow: search, customize output name, save results, use optimized model
python scripts/evaluate_profiles.py --search --limit 25 --list-name "tech-founders" --save --model-path models/founder_rank_optimized.pkl
```

## Data Directory Structure
```
data/
├── encoded/                    # Encoded founder profiles with outcomes (csv)
├── live/                       # Live YC batch scrapes, some with targets (csv)
├── proxycurl/                  # Raw LinkedIn profile data (json)
├── parsed/                     # Intermediate YC batch scrapes (csv)
├── sample_encodings/           # Example encoded profiles (csv)
├── synth/                      # Synthetic training data (csv)         
└── linkedin_profiles.json      # Linkedin profiles master json

models/                         # Trained model files
└── founder_rank.pkl            
```
## Development

For development and analysis, Jupyter notebooks are available in the `notebooks/` directory:

- `founder-rank.ipynb`: Main ranking workflow
- `data.ipynb`: Data generation and YC EDA
- `model.ipynb`: Model training and evaluation
- `live-data.ipynb`: Live data processing
- `matrix-pipeline.ipynb`: Naive untrained matrix implementation

