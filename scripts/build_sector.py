import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Out-of-band preparation of sector working data.")
    parser.add_argument("--sector", required=True, help="Sector ID (e.g. sector_001)")
    parser.add_argument("--source", required=True, help="Source to prepare (e.g. ohrc, lro_nac)")
    
    args = parser.parse_args()
    
    print(f"--- Sector Preparation for {args.sector} ---")
    print("NOTE: This is a safe preparation tool.")
    print("Real sector preparation (e.g., ISIS/PDS-aware cropping of large archives) is not yet configured.")
    print("The raw OHRC archive (1.1 GB) will NOT be extracted automatically.")
    print("The original NASA/ISRO archives at data/raw/ remain untouched as immutable sources.")
    print("")
    print("For this MVP, please use DEMO mode which utilizes pre-generated synthetic/local data.")
    print("Start the backend normally and the DataManager will automatically resolve to DEMO mode.")
    sys.exit(0)

if __name__ == "__main__":
    main()
