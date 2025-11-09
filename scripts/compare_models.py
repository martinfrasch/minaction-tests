#!/usr/bin/env python3
"""
Comparative analysis and visualization for multi-model test results
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_results(results_dir: str, pattern: str = "*.json") -> List[Dict[str, Any]]:
    """Load all result files from directory"""
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # Load all JSON files
    results = []
    for file_path in results_path.glob(pattern):
        if file_path.name.startswith('batch_summary'):
            continue  # Skip batch summary files

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if data.get('success', False):
                    results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")

    return results

def create_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison table from results"""
    rows = []

    for result in results:
        model = result['model']
        summary = result.get('summary', {})

        row = {
            'Model': model,
            'Overall Score': summary.get('overall_score', 0),
            'Passed': summary.get('passed', 0),
            'Failed': summary.get('failed', 0),
            'Partial': summary.get('partial', 0),
            'Total Tests': summary.get('total_tests', 0),
        }

        # Add category scores
        for category, score in summary.get('by_category', {}).items():
            row[f'{category}'] = score

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by overall score
    if 'Overall Score' in df.columns:
        df = df.sort_values('Overall Score', ascending=False)

    return df

def create_detailed_comparison(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create detailed test-by-test comparison"""
    test_data = {}

    for result in results:
        model = result['model']

        for test_result in result.get('results', []):
            test_name = test_result.get('test_name', 'unknown')
            evaluation = test_result.get('evaluation', {})
            score = evaluation.get('score', 0)

            if test_name not in test_data:
                test_data[test_name] = {}

            test_data[test_name][model] = score

    # Convert to DataFrame
    df = pd.DataFrame(test_data).T
    df.index.name = 'Test'

    return df

def analyze_patterns(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze patterns across models"""
    analysis = {
        'total_models': len(results),
        'average_score': 0,
        'best_model': None,
        'worst_model': None,
        'hardest_test': None,
        'easiest_test': None,
        'category_analysis': {},
        'test_analysis': {}
    }

    if not results:
        return analysis

    # Overall scores
    scores = [(r['model'], r.get('summary', {}).get('overall_score', 0)) for r in results]
    scores.sort(key=lambda x: x[1], reverse=True)

    analysis['average_score'] = sum(s[1] for s in scores) / len(scores)
    analysis['best_model'] = scores[0] if scores else None
    analysis['worst_model'] = scores[-1] if scores else None

    # Test-level analysis
    test_scores = {}
    for result in results:
        for test_result in result.get('results', []):
            test_name = test_result.get('test_name', 'unknown')
            evaluation = test_result.get('evaluation', {})
            score = evaluation.get('score', 0)

            if test_name not in test_scores:
                test_scores[test_name] = []
            test_scores[test_name].append(score)

    # Find hardest and easiest tests
    test_averages = {name: sum(scores) / len(scores) for name, scores in test_scores.items()}
    if test_averages:
        analysis['hardest_test'] = min(test_averages.items(), key=lambda x: x[1])
        analysis['easiest_test'] = max(test_averages.items(), key=lambda x: x[1])

    analysis['test_analysis'] = {
        name: {
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'std_dev': pd.Series(scores).std()
        }
        for name, scores in test_scores.items()
    }

    # Category analysis
    category_scores = {}
    for result in results:
        for category, score in result.get('summary', {}).get('by_category', {}).items():
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)

    analysis['category_analysis'] = {
        category: {
            'average_score': sum(scores) / len(scores),
            'min_score': min(scores),
            'max_score': max(scores),
            'std_dev': pd.Series(scores).std()
        }
        for category, scores in category_scores.items()
    }

    return analysis

def print_comparison(results: List[Dict[str, Any]], detailed: bool = False):
    """Print comparison tables"""
    if not results:
        print("No results to compare")
        return

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)

    # Main comparison table
    df = create_comparison_table(results)
    print("\n" + df.to_string(index=False))

    # Analysis
    analysis = analyze_patterns(results)

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    print(f"\nTotal models tested: {analysis['total_models']}")
    print(f"Average score: {analysis['average_score']:.1%}")

    if analysis['best_model']:
        print(f"\nBest performing model:")
        print(f"  {analysis['best_model'][0]}: {analysis['best_model'][1]:.1%}")

    if analysis['worst_model']:
        print(f"\nWorst performing model:")
        print(f"  {analysis['worst_model'][0]}: {analysis['worst_model'][1]:.1%}")

    if analysis['hardest_test']:
        print(f"\nHardest test (lowest average score):")
        print(f"  {analysis['hardest_test'][0]}: {analysis['hardest_test'][1]:.1%}")

    if analysis['easiest_test']:
        print(f"\nEasiest test (highest average score):")
        print(f"  {analysis['easiest_test'][0]}: {analysis['easiest_test'][1]:.1%}")

    print("\n" + "-"*80)
    print("CATEGORY ANALYSIS")
    print("-"*80)
    for category, stats in analysis['category_analysis'].items():
        print(f"\n{category}:")
        print(f"  Average: {stats['average_score']:.1%}")
        print(f"  Range: {stats['min_score']:.1%} - {stats['max_score']:.1%}")
        print(f"  Std Dev: {stats['std_dev']:.3f}")

    if detailed:
        print("\n" + "-"*80)
        print("TEST-BY-TEST ANALYSIS")
        print("-"*80)
        for test_name, stats in analysis['test_analysis'].items():
            print(f"\n{test_name}:")
            print(f"  Average: {stats['average_score']:.1%}")
            print(f"  Range: {stats['min_score']:.1%} - {stats['max_score']:.1%}")
            print(f"  Std Dev: {stats['std_dev']:.3f}")

        print("\n" + "-"*80)
        print("DETAILED TEST SCORES BY MODEL")
        print("-"*80)
        detailed_df = create_detailed_comparison(results)
        print("\n" + detailed_df.to_string())

def export_comparison(results: List[Dict[str, Any]], output_file: str, format: str = 'csv'):
    """Export comparison to file"""
    df = create_comparison_table(results)
    detailed_df = create_detailed_comparison(results)
    analysis = analyze_patterns(results)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'csv':
        # Export main table
        main_csv = output_path.parent / f"{output_path.stem}_summary.csv"
        df.to_csv(main_csv, index=False)
        print(f"Summary exported to: {main_csv}")

        # Export detailed table
        detailed_csv = output_path.parent / f"{output_path.stem}_detailed.csv"
        detailed_df.to_csv(detailed_csv)
        print(f"Detailed scores exported to: {detailed_csv}")

    elif format == 'json':
        # Export everything as JSON
        export_data = {
            'summary_table': df.to_dict(orient='records'),
            'detailed_scores': detailed_df.to_dict(),
            'analysis': analysis
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"Data exported to: {output_file}")

    elif format == 'markdown':
        # Export as markdown
        with open(output_file, 'w') as f:
            f.write("# Model Comparison Results\n\n")
            f.write("## Summary\n\n")
            f.write(df.to_markdown(index=False))

            f.write("\n\n## Analysis\n\n")
            f.write(f"- **Total models tested**: {analysis['total_models']}\n")
            f.write(f"- **Average score**: {analysis['average_score']:.1%}\n")

            if analysis['best_model']:
                f.write(f"- **Best model**: {analysis['best_model'][0]} ({analysis['best_model'][1]:.1%})\n")

            if analysis['worst_model']:
                f.write(f"- **Worst model**: {analysis['worst_model'][0]} ({analysis['worst_model'][1]:.1%})\n")

            f.write("\n## Detailed Scores\n\n")
            f.write(detailed_df.to_markdown())

        print(f"Markdown exported to: {output_file}")

def generate_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Generate visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("Warning: matplotlib/seaborn not available. Skipping visualizations.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Overall scores bar chart
    df = create_comparison_table(results)

    plt.figure(figsize=(12, 6))
    plt.barh(df['Model'], df['Overall Score'])
    plt.xlabel('Overall Score')
    plt.title('Model Performance Comparison')
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path / 'overall_scores.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'overall_scores.png'}")
    plt.close()

    # 2. Category comparison heatmap
    category_cols = [col for col in df.columns if col not in ['Model', 'Overall Score', 'Passed', 'Failed', 'Partial', 'Total Tests']]

    if category_cols:
        plt.figure(figsize=(10, len(df) * 0.5 + 2))
        category_data = df.set_index('Model')[category_cols]
        sns.heatmap(category_data, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
        plt.title('Performance by Category')
        plt.tight_layout()
        plt.savefig(output_path / 'category_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path / 'category_heatmap.png'}")
        plt.close()

    # 3. Detailed test scores heatmap
    detailed_df = create_detailed_comparison(results)

    plt.figure(figsize=(len(detailed_df.columns) * 0.8 + 2, len(detailed_df) * 0.5 + 2))
    sns.heatmap(detailed_df, annot=True, fmt='.2f', cmap='RdYlGn', vmin=0, vmax=1)
    plt.title('Detailed Test Scores')
    plt.xlabel('Model')
    plt.ylabel('Test')
    plt.tight_layout()
    plt.savefig(output_path / 'detailed_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'detailed_heatmap.png'}")
    plt.close()

    print(f"\nAll visualizations saved to: {output_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Compare results across multiple LLM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all results in directory
  python scripts/compare_models.py --input results/batch

  # Export comparison to CSV
  python scripts/compare_models.py --input results/batch --export comparison.csv

  # Generate visualizations
  python scripts/compare_models.py --input results/batch --visualize --output-dir results/viz
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input directory containing result JSON files'
    )

    parser.add_argument(
        '--export',
        type=str,
        help='Export comparison to file (format based on extension: .csv, .json, .md)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/visualizations',
        help='Output directory for visualizations'
    )

    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed test-by-test analysis'
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.input}")
    results = load_results(args.input)

    if not results:
        print("No valid results found")
        return 1

    print(f"Loaded {len(results)} result file(s)")

    # Print comparison
    print_comparison(results, detailed=args.detailed)

    # Export if requested
    if args.export:
        # Determine format from extension
        ext = Path(args.export).suffix.lower()
        if ext == '.csv':
            format = 'csv'
        elif ext == '.json':
            format = 'json'
        elif ext in ['.md', '.markdown']:
            format = 'markdown'
        else:
            print(f"Warning: Unknown format '{ext}', defaulting to CSV")
            format = 'csv'

        export_comparison(results, args.export, format)

    # Generate visualizations if requested
    if args.visualize:
        generate_visualizations(results, args.output_dir)

    return 0

if __name__ == '__main__':
    sys.exit(main())
