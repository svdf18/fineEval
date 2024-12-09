import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_visualizations():
    # Set up paths
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Read the results
    print("Loading evaluation results...")
    df = pd.read_csv(docs_dir / 'evaluation_results.csv')
    
    # Set style
    plt.style.use('default')
    sns.set_theme()
    
    # Create visualizations directory
    viz_dir = docs_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Response Time Comparison
    print("Creating response time visualization...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='response_time', data=df)
    plt.title('Response Time by Model')
    plt.ylabel('Time (seconds)')
    plt.savefig(viz_dir / 'response_times.png')
    plt.close()
    
    # 2. Response Length Comparison
    print("Creating response length visualization...")
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='model', y='response_length', data=df)
    plt.title('Response Length by Model')
    plt.ylabel('Length (characters)')
    plt.savefig(viz_dir / 'response_lengths.png')
    plt.close()
    
    # 3. Time vs Length Scatter Plot
    print("Creating time vs length scatter plot...")
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='response_length', y='response_time', hue='model', style='model')
    plt.title('Response Time vs Length')
    plt.xlabel('Response Length (characters)')
    plt.ylabel('Response Time (seconds)')
    plt.savefig(viz_dir / 'time_vs_length.png')
    plt.close()
    
    # 4. Response Time per Question Type
    print("Creating question type analysis...")
    plt.figure(figsize=(15, 8))
    df['question_type'] = df['prompt'].apply(lambda x: x.split()[0])  # Simple way to categorize
    sns.barplot(x='question_type', y='response_time', hue='model', data=df)
    plt.title('Response Time by Question Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(viz_dir / 'question_type_analysis.png')
    plt.close()
    
    print(f"\nVisualizations have been saved in {viz_dir}/")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary = df.groupby('model').agg({
        'response_time': ['mean', 'std', 'min', 'max'],
        'response_length': ['mean', 'std', 'min', 'max']
    }).round(2)
    print(summary)

if __name__ == "__main__":
    create_visualizations()
