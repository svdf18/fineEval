import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from textblob import TextBlob
import spacy
import numpy as np
from pathlib import Path
import json

class QualityAnalyzer:
    def __init__(self):
        print("Initializing Quality Analyzer...")
        # Create docs directory if it doesn't exist
        self.docs_dir = Path("docs")
        self.docs_dir.mkdir(exist_ok=True)
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            self.nlp = spacy.load('en_core_web_sm')
        except:
            print("Please install required packages:")
            print("pip install nltk textblob spacy")
            print("python -m spacy download en_core_web_sm")
            raise

    def analyze_responses(self, df):
        print("Analyzing response quality...")
        metrics = []
        
        for _, row in df.iterrows():
            response = row['response']
            
            # Basic metrics
            sentences = sent_tokenize(response)
            words = word_tokenize(response)
            
            # Calculate metrics
            metrics.append({
                'model': row['model'],
                'prompt': row['prompt'],
                'num_sentences': len(sentences),
                'num_words': len(words),
                'avg_sentence_length': len(words) / len(sentences),
                'readability_score': self._calculate_readability(response),
                'code_blocks': self._count_code_blocks(response),
                'structure_score': self._evaluate_structure(response),
                'specificity_score': self._calculate_specificity(response)
            })
        
        return pd.DataFrame(metrics)

    def _calculate_readability(self, text):
        """Calculate Flesch Reading Ease score"""
        return TextBlob(text).sentiment.polarity

    def _count_code_blocks(self, text):
        """Count number of code blocks in response"""
        return text.count('```')

    def _evaluate_structure(self, text):
        """Evaluate structural elements (lists, sections, etc.)"""
        score = 0
        score += text.count('*') * 0.5  # Bullet points
        score += text.count('#') * 1.0  # Headers
        score += text.count('\n\n') * 0.3  # Paragraphs
        return score

    def _calculate_specificity(self, text):
        """Calculate specificity based on technical terms and concrete examples"""
        doc = self.nlp(text)
        technical_terms = len([token for token in doc if token.pos_ in ['NOUN', 'PROPN']])
        return technical_terms / len(doc)

def main():
    # Set up paths
    docs_dir = Path("docs")
    
    # Load results
    print("Loading evaluation results...")
    df = pd.read_csv(docs_dir / 'evaluation_results.csv')
    
    # Create analyzer
    analyzer = QualityAnalyzer()
    
    # Get quality metrics
    quality_df = analyzer.analyze_responses(df)
    
    # Save detailed metrics
    quality_df.to_csv(docs_dir / 'quality_metrics.csv', index=False)
    
    # Calculate summary
    summary = quality_df.groupby('model').agg({
        'num_sentences': ['mean', 'std'],
        'num_words': ['mean', 'std'],
        'readability_score': ['mean', 'std'],
        'code_blocks': ['mean', 'sum'],
        'structure_score': ['mean', 'std'],
        'specificity_score': ['mean', 'std']
    }).round(2)
    
    # Convert to a more JSON-friendly format
    summary_dict = {}
    for model in summary.index:
        summary_dict[model] = {}
        for col in summary.columns:
            metric, stat = col
            if metric not in summary_dict[model]:
                summary_dict[model][metric] = {}
            summary_dict[model][metric][stat] = float(summary.loc[model, col])
    
    # Save summary as JSON
    with open(docs_dir / 'quality_summary.json', 'w') as f:
        json.dump(summary_dict, f, indent=2)
    
    # Print summary in a readable format
    print("\nQuality Metrics Summary:")
    print(summary)
    
    # Compare models
    print("\nStatistical Comparison:")
    for metric in ['num_sentences', 'num_words', 'readability_score', 'structure_score', 'specificity_score']:
        base = quality_df[quality_df['model'] == 'base'][metric]
        finetuned = quality_df[quality_df['model'] == 'finetuned'][metric]
        
        comparison = {
            'metric': metric,
            'base_mean': float(base.mean()),
            'finetuned_mean': float(finetuned.mean()),
            'difference_percent': float((finetuned.mean() - base.mean()) / base.mean() * 100)
        }
        
        print(f"\n{metric}:")
        print(f"Base model mean: {comparison['base_mean']:.2f}")
        print(f"Finetuned model mean: {comparison['finetuned_mean']:.2f}")
        print(f"Difference: {comparison['difference_percent']:.1f}%")
        
        # Save comparison data
        with open(docs_dir / f'comparison_{metric}.json', 'w') as f:
            json.dump(comparison, f, indent=2)

if __name__ == "__main__":
    main() 