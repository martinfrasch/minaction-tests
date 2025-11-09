#!/usr/bin/env python3
"""
Batch testing script for running tests across multiple LLM architectures
"""

import argparse
import json
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_interface import get_model_interface
from src.evaluation import evaluate_response, aggregate_results
from tests import (
    test_forward_derivation,
    test_inverse_problems,
    test_physical_constraints,
    test_cross_domain,
    test_symmetry
)

# Test suites (same as run_tests.py)
TEST_SUITES = {
    'complete': [
        ('forward_general', test_forward_derivation.test_general_lagrangian),
        ('forward_harmonic', test_forward_derivation.test_harmonic_oscillator),
        ('forward_novel_1', test_forward_derivation.test_novel_lagrangian_1),
        ('forward_novel_2', test_forward_derivation.test_novel_lagrangian_2),
        ('inverse_find_lagrangian', test_inverse_problems.test_find_lagrangian),
        ('constraints_invalid', test_physical_constraints.test_invalid_lagrangian),
        ('constraints_higher_derivatives', test_physical_constraints.test_higher_derivatives),
        ('cross_population', test_cross_domain.test_population_dynamics),
        ('symmetry_noether', test_symmetry.test_rotational_symmetry)
    ],
    'forward_only': [
        ('forward_general', test_forward_derivation.test_general_lagrangian),
        ('forward_harmonic', test_forward_derivation.test_harmonic_oscillator),
        ('forward_novel_1', test_forward_derivation.test_novel_lagrangian_1),
        ('forward_novel_2', test_forward_derivation.test_novel_lagrangian_2)
    ],
    'understanding_only': [
        ('inverse_find_lagrangian', test_inverse_problems.test_find_lagrangian),
        ('constraints_invalid', test_physical_constraints.test_invalid_lagrangian),
        ('constraints_higher_derivatives', test_physical_constraints.test_higher_derivatives)
    ],
}

def load_model_configs(config_path: str = None) -> Dict[str, Any]:
    """Load model configurations from YAML file"""
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__),
            '..',
            'config',
            'model_configs.yaml'
        )

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_models_from_config(config_name: str, configs: Dict[str, Any]) -> List[str]:
    """Get list of models from a test configuration"""
    if config_name not in configs.get('test_configs', {}):
        raise ValueError(f"Unknown test configuration: {config_name}")

    config = configs['test_configs'][config_name]

    # Handle 'all_available' special case
    if config_name == 'all_available':
        models = []
        for provider in configs['models'].values():
            models.extend([m['name'] for m in provider])
        return models

    return config['models']

def run_test_for_model(model_name: str, test_suite: str, verbose: bool = False) -> Dict[str, Any]:
    """Run tests for a single model"""

    print(f"\n{'='*70}")
    print(f"Testing Model: {model_name}")
    print(f"Test Suite: {test_suite}")
    print('='*70)

    try:
        # Initialize model interface
        model = get_model_interface(model_name)

        # Get test suite
        if test_suite not in TEST_SUITES:
            raise ValueError(f"Unknown test suite: {test_suite}")

        tests = TEST_SUITES[test_suite]

        # Run all tests
        results = []
        for i, (test_name, test_func) in enumerate(tests, 1):
            print(f"\n[{i}/{len(tests)}] Running: {test_name}")

            try:
                # Run the test
                result = test_func(model)

                # Evaluate the response
                evaluation = evaluate_response(result)

                results.append({
                    'test_name': test_name,
                    'result': result,
                    'evaluation': evaluation,
                    'timestamp': datetime.now().isoformat()
                })

                # Print status
                status = '✅' if evaluation['passed'] else '❌'
                print(f"  {status} Score: {evaluation['score']:.2f}")

                if verbose and not evaluation['passed']:
                    print(f"  Response preview: {result['response'][:200]}...")

            except Exception as e:
                print(f"  ❌ Test failed with error: {str(e)}")
                if verbose:
                    traceback.print_exc()
                results.append({
                    'test_name': test_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })

        # Aggregate results
        summary = aggregate_results(results)

        # Print summary
        print(f"\n{'-'*70}")
        print(f"Model: {model_name}")
        print(f"Overall Score: {summary['overall_score']:.1%}")
        print(f"Passed: {summary['passed']}/{summary['total_tests']}")
        print('-'*70)

        return {
            'model': model_name,
            'suite': test_suite,
            'summary': summary,
            'results': results,
            'success': True,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"\n❌ Model testing failed: {str(e)}")
        if verbose:
            traceback.print_exc()

        return {
            'model': model_name,
            'suite': test_suite,
            'error': str(e),
            'success': False,
            'timestamp': datetime.now().isoformat()
        }

def run_batch_tests(
    models: List[str],
    test_suite: str = 'complete',
    output_dir: str = 'results/batch',
    verbose: bool = False,
    continue_on_error: bool = True
) -> List[Dict[str, Any]]:
    """Run tests for multiple models"""

    print(f"\n{'='*70}")
    print(f"BATCH TESTING")
    print(f"Models: {len(models)}")
    print(f"Suite: {test_suite}")
    print('='*70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run tests for each model
    all_results = []
    successful_models = []
    failed_models = []

    for i, model_name in enumerate(models, 1):
        print(f"\n{'#'*70}")
        print(f"Model {i}/{len(models)}")
        print('#'*70)

        result = run_test_for_model(model_name, test_suite, verbose)
        all_results.append(result)

        if result['success']:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
            if not continue_on_error:
                print("\n⚠️  Stopping batch tests due to error (--continue-on-error not set)")
                break

    # Save individual results
    for result in all_results:
        if result['success']:
            model_name_safe = result['model'].replace('/', '_').replace(':', '_')
            result_file = output_path / f"{model_name_safe}_{test_suite}.json"

            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

    # Save batch summary
    batch_summary = {
        'test_suite': test_suite,
        'total_models': len(models),
        'successful_models': len(successful_models),
        'failed_models': len(failed_models),
        'models_tested': successful_models,
        'models_failed': failed_models,
        'timestamp': datetime.now().isoformat(),
        'results': all_results
    }

    summary_file = output_path / f"batch_summary_{test_suite}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2, default=str)

    # Print final summary
    print(f"\n\n{'='*70}")
    print("BATCH TEST SUMMARY")
    print('='*70)
    print(f"Total models: {len(models)}")
    print(f"Successful: {len(successful_models)} ✅")
    print(f"Failed: {len(failed_models)} ❌")

    if successful_models:
        print(f"\nSuccessful models:")
        for model in successful_models:
            # Find result for this model
            model_result = next(r for r in all_results if r['model'] == model)
            score = model_result.get('summary', {}).get('overall_score', 0)
            print(f"  • {model}: {score:.1%}")

    if failed_models:
        print(f"\nFailed models:")
        for model in failed_models:
            print(f"  • {model}")

    print(f"\nResults saved to: {output_path}")
    print(f"Summary saved to: {summary_file}")
    print('='*70)

    return all_results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run batch tests across multiple LLM architectures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test frontier models
  python scripts/run_batch_tests.py --config frontier_comparison

  # Test specific models
  python scripts/run_batch_tests.py --models gpt-4o claude-3-5-sonnet-20241022 llama3.1:70b

  # Test open-source models on forward derivation only
  python scripts/run_batch_tests.py --config open_source_comparison --suite forward_only

  # List available configurations
  python scripts/run_batch_tests.py --list-configs
        """
    )

    parser.add_argument(
        '--models',
        nargs='+',
        help='Specific models to test'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Test configuration name from model_configs.yaml'
    )

    parser.add_argument(
        '--suite',
        type=str,
        default='complete',
        choices=list(TEST_SUITES.keys()),
        help='Test suite to run (default: complete)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/batch',
        help='Output directory for results'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        default=True,
        help='Continue testing other models if one fails (default: true)'
    )

    parser.add_argument(
        '--list-configs',
        action='store_true',
        help='List available test configurations'
    )

    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to custom model_configs.yaml file'
    )

    args = parser.parse_args()

    # Load model configurations
    configs = load_model_configs(args.config_file)

    # List configurations if requested
    if args.list_configs:
        print("\nAvailable Test Configurations:")
        print("="*70)
        for name, config in configs.get('test_configs', {}).items():
            print(f"\n{name}:")
            print(f"  Description: {config['description']}")
            if config['models']:
                print(f"  Models ({len(config['models'])}):")
                for model in config['models']:
                    print(f"    - {model}")
        return 0

    # Determine which models to test
    models = []
    if args.models:
        models = args.models
    elif args.config:
        models = get_models_from_config(args.config, configs)
    else:
        parser.error("Must specify either --models or --config")

    if not models:
        parser.error("No models specified")

    # Run batch tests
    results = run_batch_tests(
        models=models,
        test_suite=args.suite,
        output_dir=args.output,
        verbose=args.verbose,
        continue_on_error=args.continue_on_error
    )

    # Return non-zero if any model failed
    if any(not r['success'] for r in results):
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
