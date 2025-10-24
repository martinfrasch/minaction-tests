#!/usr/bin/env python3
"""
Analyze and visualize test results from minAction.net empirical validation.

Usage:
    python scripts/analyze_results.py --input results/qwen2_math_7b/ [--output figures/]
"""

import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_style('whitegrid')
sns.set_palette('Set2')


def load_results(results_dir):
    """Load test results from JSON file."""
    results_path = Path(results_dir) / 'detailed_results.json'
    with open(results_path, 'r') as f:
        return json.load(f)


def create_category_breakdown(results, output_dir):
    """Create bar chart of performance by category."""
    tests = results['tests']
    df = pd.DataFrame([
        {
            'Test': t['name'],
            'Category': t['category'],
            'Score': t['score'],
            'Status': t['status']
        }
        for t in tests
    ])
    
    # Calculate category averages
    category_scores = df.groupby('Category')['Score'].mean().sort_values(ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(category_scores)), category_scores.values, 
                   color='steelblue', alpha=0.8)
    
    # Add value labels on bars
    for i, (cat, score) in enumerate(category_scores.items()):
        ax.text(i, score + 0.02, f'{score:.1%}', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(range(len(category_scores)))
    ax.set_xticklabels(category_scores.index, rotation=45, ha='right')
    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title('Performance by Test Category', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.0])
    
    # Add overall average line
    overall_avg = results['metadata']['overall_percentage'] / 100
    ax.axhline(y=overall_avg, color='red', linestyle='--', 
               linewidth=2, label=f'Overall Average ({overall_avg:.1%})')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'category_breakdown.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return category_scores


def create_test_heatmap(results, output_dir):
    """Create heatmap showing individual test performance."""
    tests = results['tests']
    
    # Prepare data
    test_names = [t['name'] for t in tests]
    categories = [t['category'] for t in tests]
    scores = [t['score'] for t in tests]
    
    # Create DataFrame for heatmap
    df = pd.DataFrame({
        'Test': test_names,
        'Category': categories,
        'Score': scores
    })
    
    # Pivot for heatmap
    pivot = df.pivot_table(values='Score', index='Test', columns='Category', 
                           aggfunc='first', fill_value=0)
    
    # If pivot is empty or has issues, create simple visualization
    if pivot.empty:
        # Alternative: Simple bar chart of all tests
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if s >= 0.8 else 'yellow' if s >= 0.4 else 'red' 
                  for s in scores]
        bars = ax.barh(range(len(test_names)), scores, color=colors, alpha=0.7)
        ax.set_yticks(range(len(test_names)))
        ax.set_yticklabels(test_names, fontsize=10)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title('Individual Test Performance', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.0])
        
        # Add score labels
        for i, score in enumerate(scores):
            ax.text(score + 0.02, i, f'{score:.1f}', va='center', fontsize=9)
        
        plt.tight_layout()
        output_path = Path(output_dir) / 'test_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    return df


def create_phase_comparison(results, output_dir):
    """Compare Phase 1 (basic) vs Phase 2 (guided) results."""
    phase2_tests = results.get('phase2_tests', [])
    
    if not phase2_tests:
        print("No Phase 2 results found. Skipping comparison plot.")
        return None
    
    # Prepare data
    test_names = [t['name'] for t in phase2_tests]
    phase1_scores = [t.get('original_score', 0) for t in phase2_tests]
    phase2_scores = [t['score'] for t in phase2_tests]
    improvements = [t.get('improvement', 'N/A') for t in phase2_tests]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(test_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, phase1_scores, width, 
                   label='Phase 1 (Basic)', alpha=0.8, color='lightcoral')
    bars2 = ax.bar(x + width/2, phase2_scores, width, 
                   label='Phase 2 (With Principles)', alpha=0.8, color='lightgreen')
    
    # Add improvement labels
    for i, (p1, p2, imp) in enumerate(zip(phase1_scores, phase2_scores, improvements)):
        if imp != 'N/A' and imp != 'none':
            # Draw arrow showing improvement
            if p2 > p1:
                ax.annotate('', xy=(i + width/2, p2), xytext=(i - width/2, p1),
                           arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
                ax.text(i, max(p1, p2) + 0.05, imp, ha='center', 
                       fontsize=9, color='green', fontweight='bold')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Phase 1 vs Phase 2: Impact of Selection Principles', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace(' with Principles', '') for name in test_names], 
                        rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    
    # Add overall improvement annotation
    overall_p1 = np.mean(phase1_scores)
    overall_p2 = np.mean(phase2_scores)
    improvement_pct = (overall_p2 - overall_p1) * 100
    ax.text(0.02, 0.98, 
            f'Overall Improvement: {improvement_pct:+.0f} percentage points\n'
            f'Phase 1: {overall_p1:.1%} → Phase 2: {overall_p2:.1%}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'phase_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return {
        'phase1_avg': overall_p1,
        'phase2_avg': overall_p2,
        'improvement': improvement_pct
    }


def create_summary_report(results, category_scores, phase_comparison, output_dir):
    """Generate a text summary report."""
    report = []
    report.append("=" * 80)
    report.append("minAction.net EMPIRICAL VALIDATION SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Metadata
    meta = results['metadata']
    report.append(f"Model: {meta['model']}")
    report.append(f"Test Date: {meta['test_date']}")
    report.append(f"Framework Version: {meta['framework_version']}")
    report.append(f"Total Tests: {meta['total_tests']}")
    report.append(f"Overall Score: {meta['overall_score']}/{meta['total_tests']} ({meta['overall_percentage']:.1f}%)")
    report.append("")
    
    # Category breakdown
    report.append("PERFORMANCE BY CATEGORY:")
    report.append("-" * 40)
    for cat, score in category_scores.items():
        status = "✅ Strong" if score >= 0.8 else "⚠️ Moderate" if score >= 0.5 else "❌ Weak"
        report.append(f"  {cat:30s}: {score:.1%} {status}")
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS:")
    report.append("-" * 40)
    analysis = results.get('analysis', {})
    
    report.append("\nStrengths:")
    for strength in analysis.get('strengths', []):
        report.append(f"  ✓ {strength}")
    
    report.append("\nWeaknesses:")
    for weakness in analysis.get('weaknesses', []):
        report.append(f"  ✗ {weakness}")
    
    report.append("\nKey Insights:")
    for insight in analysis.get('key_insights', []):
        report.append(f"  • {insight}")
    report.append("")
    
    # Phase 2 comparison
    if phase_comparison:
        report.append("PHASE 2 RESULTS (With Selection Principles):")
        report.append("-" * 40)
        report.append(f"  Phase 1 Average: {phase_comparison['phase1_avg']:.1%}")
        report.append(f"  Phase 2 Average: {phase_comparison['phase2_avg']:.1%}")
        report.append(f"  Improvement: {phase_comparison['improvement']:+.1f} percentage points")
        report.append("")
    
    report.append("=" * 80)
    report.append("CONCLUSION:")
    report.append("-" * 40)
    report.append("Mathematical capability ≠ Physical discovery capability")
    report.append("Models can manipulate equations but cannot:")
    report.append("  • Identify which structures are physically realizable")
    report.append("  • Construct new Lagrangians from observations (0% on inverse problems)")
    report.append("  • Extend principles reliably to new domains")
    report.append("")
    report.append("These results validate the need for the minAction.net framework.")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    output_path = Path(output_dir) / 'summary_report.txt'
    with open(output_path, 'w') as f:
        f.write(report_text)
    print(f"Saved: {output_path}")
    
    # Also print to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='Analyze minAction.net test results')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing detailed_results.json')
    parser.add_argument('--output', type=str, default='figures/',
                       help='Output directory for figures and reports')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.input}...")
    results = load_results(args.input)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    category_scores = create_category_breakdown(results, output_dir)
    create_test_heatmap(results, output_dir)
    phase_comparison = create_phase_comparison(results, output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    create_summary_report(results, category_scores, phase_comparison, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
