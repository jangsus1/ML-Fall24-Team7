# Detecting User Actions from Mouse Events

## Link to Reports
[Proposal](docs/PROPOSAL.md)
[Mid-term Report](docs/MID_CHECKPOINT.md)


## Project Structure

```plaintext
├── README.md                   # Main project documentation
├── features.py                 # Function for feature extraction
├── feature_extraction.ipynb    # Feature Extracting & Training code
├── recorder.ipynb              # Recording code for user mouse events
├── live_inference.ipynb        # Live inference demo code (preparing)
├── recordings/                 # Recorded streams of user actions
│   └── *.txt                   # Stream of action with 5-20 minute length
├── train_data/                 # Extracted features from the recorded stream
│   └── *.csv                   # Extracted features per recording
├── docs/                       # Documentation directory
|   ├── PROPOSAL.md             # Proposal report
│   └── MID_CHECKPOINT.md       # Mid term report
├── assets/                     # Image files for reports
│   └── *.png                   # Corresponding image files
└── requirements.txt            # Python dependencies - May ignore specific versions
```