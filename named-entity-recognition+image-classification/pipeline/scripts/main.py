import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import argparse
from pipeline.src.evaluation import test_evaluation_data, interactive_mode
from pipeline.src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline for animal recognition.")
    parser.add_argument("--text", type=str, help="Input text message.")
    parser.add_argument("--image", type=str, help="Path to the input image.")
    args = parser.parse_args()
    
    if args.text and args.image:
        result = run_pipeline(args.text, args.image)
        print(f"Pipeline result: {result}")
    else:
        print("No --text or --image arguments provided. Running tests and interactive mode.\n")
        test_evaluation_data()
        interactive_mode()

if __name__ == "__main__":
    main()