import requests
import json
import time
import pandas as pd
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Create docs directory if it doesn't exist
docs_dir = Path("docs")
docs_dir.mkdir(exist_ok=True)

class ModelEvaluator:
    def __init__(self):
        self.base_url = os.getenv("BASE_URL")
        self.models = {
            'base': os.getenv("BASE_MODEL"),
            'finetuned': os.getenv("FINETUNED_MODEL")
        }
        print(f"Initialized evaluator with models: {list(self.models.values())}")
        print(f"Using API endpoint: {self.base_url}")

    def query_model(self, prompt: str, model_name: str) -> Tuple[str, float]:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        
        start_time = time.time()
        response = requests.post(self.base_url, json=payload)
        end_time = time.time()
        
        if response.status_code != 200:
            raise Exception(f"API call failed with status {response.status_code}: {response.text}")
            
        return response.json()['response'], end_time - start_time

    def evaluate_models(self, test_cases: List[Dict]) -> pd.DataFrame:
        results = []
        total_cases = len(test_cases)
        
        print(f"\nStarting evaluation of {total_cases} test cases across {len(self.models)} models...")
        
        for idx, case in enumerate(test_cases, 1):
            prompt = case['prompt']
            expected = case.get('expected', None)
            
            print(f"\nCase {idx}/{total_cases}: {prompt[:50]}...")
            
            for model_key, model_name in self.models.items():
                print(f"  Testing {model_key}...", end=' ', flush=True)
                try:
                    response, response_time = self.query_model(prompt, model_name)
                    print(f"Done ({response_time:.2f}s)")
                    
                    result = {
                        'model': model_key,
                        'prompt': prompt,
                        'response': response,
                        'expected': expected,
                        'response_time': response_time,
                        'response_length': len(response)
                    }
                    results.append(result)
                except Exception as e:
                    print(f"Error: {str(e)}")
                    print(f"Skipping {model_key} for this case.")
        
        print("\nEvaluation complete!")
        return pd.DataFrame(results)

    def calculate_metrics(self, df: pd.DataFrame) -> Dict:
        metrics = {}
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            
            metrics[model] = {
                'avg_response_time': model_df['response_time'].mean(),
                'avg_response_length': model_df['response_length'].mean(),
                'total_responses': len(model_df),
                'min_response_time': model_df['response_time'].min(),
                'max_response_time': model_df['response_time'].max(),
                'std_response_time': model_df['response_time'].std()
            }
        
        return metrics

def main():
    from test_cases import TEST_CASES
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = ModelEvaluator()
    
    # Run evaluation
    print("\nStarting evaluation...")
    results_df = evaluator.evaluate_models(TEST_CASES)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = evaluator.calculate_metrics(results_df)
    
    # Save results to docs directory
    print("\nSaving results...")
    results_path = docs_dir / 'evaluation_results.csv'
    metrics_path = docs_dir / 'metrics.json'
    
    results_df.to_csv(results_path, index=False)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Print sample comparisons
    print("\nSample Response Comparisons:")
    for prompt in results_df['prompt'].unique():
        print(f"\nPrompt: {prompt}")
        prompt_df = results_df[results_df['prompt'] == prompt]
        for _, row in prompt_df.iterrows():
            print(f"\n{row['model']} response ({row['response_time']:.2f}s):")
            print(row['response'])
            print("-" * 80)

if __name__ == "__main__":
    main() 