"""
Visualization utilities for minAction.net test results and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
import pandas as pd

sns.set_style('whitegrid')
sns.set_palette('husl')


class ActionLandscapePlotter:
    """Visualize action landscapes for physics discovery."""
    
    def __init__(self, figsize=(12, 8)):
        """Initialize plotter with default figure size."""
        self.figsize = figsize
    
    def plot_action_landscape_2d(self,
                                 action_fn: callable,
                                 x_range: tuple,
                                 y_range: tuple,
                                 true_path: Optional[np.ndarray] = None,
                                 title: str = "Action Landscape"):
        """
        Plot 2D action landscape showing minima.
        
        Args:
            action_fn: Function that computes action for (x, y) coordinates
            x_range: (min, max) for x-axis
            y_range: (min, max) for y-axis  
            true_path: Optional array of true minimum-action path
            title: Plot title
        """
        # Create grid
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        
        # Compute action at each point
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = action_fn(X[i, j], Y[i, j])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot contours
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        contour_lines = ax.contour(X, Y, Z, levels=10, colors='white', 
                                   alpha=0.4, linewidths=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Action S', fontsize=12)
        
        # Plot true path if provided
        if true_path is not None:
            ax.plot(true_path[:, 0], true_path[:, 1], 'r-', 
                   linewidth=3, label='True Minimum Path', zorder=10)
            ax.plot(true_path[0, 0], true_path[0, 1], 'go', 
                   markersize=10, label='Start', zorder=11)
            ax.plot(true_path[-1, 0], true_path[-1, 1], 'ro', 
                   markersize=10, label='End', zorder=11)
        
        ax.set_xlabel('Position (q)', fontsize=12)
        ax.set_ylabel('Velocity (q̇)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        if true_path is not None:
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_action_histogram(self,
                             perturbation_results: Dict,
                             title: str = "Action Distribution"):
        """
        Plot histogram of actions for perturbed paths.
        
        Args:
            perturbation_results: Results from SelectionPrinciples.evaluate_action_principle
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Extract data
        true_action = perturbation_results['action']
        perturbed_actions = perturbation_results['perturbation_actions']
        percentile = perturbation_results['percentile']
        
        # Plot histogram
        ax.hist(perturbed_actions, bins=50, alpha=0.7, color='steelblue',
               edgecolor='black', label='Perturbed Paths')
        
        # Mark true action
        ax.axvline(true_action, color='red', linestyle='--', linewidth=3,
                  label=f'True Path (Action = {true_action:.4f})')
        
        # Add percentile annotation
        ax.text(0.95, 0.95, 
               f'Percentile: {percentile:.1f}%\n'
               f'{"✅ Minimum!" if percentile < 5 else "❌ Not minimum"}',
               transform=ax.transAxes,
               fontsize=12, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('Action S', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax


class TestResultsVisualizer:
    """Visualize test results and performance metrics."""
    
    def __init__(self, figsize=(12, 8)):
        """Initialize visualizer."""
        self.figsize = figsize
    
    def plot_test_scores(self, 
                        test_results: List[Dict],
                        title: str = "Test Performance"):
        """
        Plot scores for individual tests.
        
        Args:
            test_results: List of test result dictionaries
            title: Plot title
        """
        # Prepare data
        test_names = [t['name'] for t in test_results]
        scores = [t['score'] for t in test_results]
        categories = [t.get('category', 'Unknown') for t in test_results]
        
        # Color by category
        unique_categories = list(set(categories))
        category_colors = {cat: sns.color_palette()[i % 10] 
                          for i, cat in enumerate(unique_categories)}
        colors = [category_colors[cat] for cat in categories]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        bars = ax.barh(range(len(test_names)), scores, color=colors, alpha=0.7)
        
        # Add score labels
        for i, (score, bar) in enumerate(zip(scores, bars)):
            status = '✅' if score >= 0.8 else '⚠️' if score >= 0.4 else '❌'
            ax.text(score + 0.02, i, f'{status} {score:.2f}', 
                   va='center', fontsize=10)
        
        ax.set_yticks(range(len(test_names)))
        ax.set_yticklabels(test_names, fontsize=10)
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1.1])
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add legend for categories
        legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=category_colors[cat], 
                          label=cat) for cat in unique_categories]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_category_radar(self,
                           category_scores: Dict[str, float],
                           title: str = "Performance Radar"):
        """
        Create radar chart of performance by category.
        
        Args:
            category_scores: Dictionary mapping category names to average scores
            title: Plot title
        """
        categories = list(category_scores.keys())
        scores = list(category_scores.values())
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        scores += scores[:1]  # Complete the circle
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, scores, 'o-', linewidth=2, label='Performance', color='steelblue')
        ax.fill(angles, scores, alpha=0.25, color='steelblue')
        
        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=9)
        ax.grid(True)
        
        # Add title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add performance zones
        ax.fill_between(angles, 0, 0.5, alpha=0.1, color='red', label='Weak (<50%)')
        ax.fill_between(angles, 0.5, 0.8, alpha=0.1, color='yellow', label='Moderate (50-80%)')
        ax.fill_between(angles, 0.8, 1.0, alpha=0.1, color='green', label='Strong (>80%)')
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_convergence(self,
                        iterations: List[int],
                        losses: List[float],
                        title: str = "Training Convergence"):
        """
        Plot training convergence curve.
        
        Args:
            iterations: List of iteration numbers
            losses: List of loss values
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(iterations, losses, 'b-', linewidth=2, alpha=0.7)
        ax.scatter(iterations[::max(1, len(iterations)//20)], 
                  losses[::max(1, len(iterations)//20)],
                  c='red', s=50, zorder=5, alpha=0.6)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Add annotation for final loss
        ax.text(0.95, 0.95,
               f'Final Loss: {losses[-1]:.6f}',
               transform=ax.transAxes,
               fontsize=11, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        plt.tight_layout()
        return fig, ax


# Example usage
if __name__ == '__main__':
    print("Visualization Module - Example Usage")
    print("=" * 60)
    
    # Example 1: Action landscape
    print("\n1. Creating action landscape plot...")
    
    def simple_action(x, v):
        """Simple quadratic action."""
        return 0.5 * (x**2 + v**2)
    
    plotter = ActionLandscapePlotter(figsize=(10, 8))
    fig, ax = plotter.plot_action_landscape_2d(
        simple_action,
        x_range=(-2, 2),
        y_range=(-2, 2),
        true_path=np.array([[0, 0], [0.5, 0.5], [1, 1]]),
        title="Example: Harmonic Oscillator Action Landscape"
    )
    plt.savefig('/tmp/example_action_landscape.png', dpi=150, bbox_inches='tight')
    print("   Saved: /tmp/example_action_landscape.png")
    plt.close()
    
    # Example 2: Test results
    print("\n2. Creating test results plot...")
    
    example_results = [
        {'name': 'Forward Derivation', 'score': 0.87, 'category': 'Forward'},
        {'name': 'Inverse Problem', 'score': 0.0, 'category': 'Inverse'},
        {'name': 'Constraint Check', 'score': 0.25, 'category': 'Constraints'},
        {'name': "Noether's Theorem", 'score': 1.0, 'category': 'Symmetry'}
    ]
    
    viz = TestResultsVisualizer(figsize=(10, 6))
    fig, ax = viz.plot_test_scores(example_results)
    plt.savefig('/tmp/example_test_scores.png', dpi=150, bbox_inches='tight')
    print("   Saved: /tmp/example_test_scores.png")
    plt.close()
    
    print("\n✅ Examples generated successfully!")
