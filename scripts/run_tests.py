#!/usr/bin/env python3
"""
Main test runner for minAction.net LLM physics understanding tests
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import sys

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

# Test suites
TEST_SUITES = {
    'complete': {
        'description': 'All 9 tests from the paper',
        'tests': [
            ('forward_general', test_forward_derivation.test_general_lagrangian),
            ('forward_harmonic', test_forward_derivation.test_harmonic_oscillator),
            ('forward_novel_1', test_forward_derivation.test_novel_lagrangian_1),
            ('forward_novel_2', test_forward_derivation.test_novel_lagrangian_2),
            ('inverse_find_lagrangian', test_inverse_problems.test_find_lagrangian),
            ('constraints_invalid', test_physical_constraints.test_invalid_lagrangian),
            ('constraints_higher_derivatives', test_physical_constraints.test_higher_derivatives),
            ('cross_population', test_cross_domain.test_population_dynamics),
            ('symmetry_noether', test_symmetry.test_rotational_symmetry)
        ]
    },
    'forward_only': {
        'description': 'Tests 1-4: Forward Euler-Lagrange application',
        'tests': [
            ('forward_general', test_forward_derivation.test_general_lagrangian),
            ('forward_harmonic', test_forward_derivation.test_harmonic_oscillator),
            ('forward_novel_1', test_forward_derivation.test_novel_lagrangian_1),
            ('forward_novel_2', test_forward_derivation.test_novel_lagrangian_2)
        ]
    },
    'understanding_only': {
        'description': 'Tests that require physical understanding',
        'tests': [
            ('inverse_find_lagrangian', test_inverse_problems.test_find_lagrangian),
            ('constraints_invalid', test_physical_constraints.test_invalid_lagrangian),
            ('constraints_higher_derivatives', test_physical_constraints.test_higher_derivatives)
        ]
    },
    'exact_reproduction': {
        'description': 'Exact tests from our paper',
        'tests': [
            ('test_1', test_forward_derivation.test_general_lagrangian),
            ('test_2', test_forward_derivation.test_harmonic_oscillator),
            ('test_3', test_forward_derivation.test_novel_lagrangian_1),
            ('test_4', test_forward_derivation.test_novel_lagrangian_2),
            ('test_5', test_inverse_problems.test_find_lagrangian),
            ('test_6', test_physical_constraints.test_invalid_lagrangian),
            ('test_7', test_physical_constraints.test_higher_derivatives),
            ('test_8', test_cross_domain.test_population_dynamics),
            ('test_9', test_symmetry.test_rotational_symmetry)
        ]
    }
}

def run_single_test(model, test_name, test_func, verbose=False):
    """Run a single test and return results"""
    
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print('='*60)
    
    # Run the test
    result = test_func(model)
    
    # Evaluate the response
    evaluation = evaluate_response(result)
    
    if verbose:
        print(f"\nPrompt: {result['prompt'][:200]}...")
        print(f"\nResponse: {result['response'][:500]}...")
        print(f"\nPassed: {'✅' if evaluation['passed'] else '❌'}")
        print(f"Score: {evaluation['score']:.2f}")
    
    return {
        'test_name': test_name,
        'result': result,
        'evaluation': evaluation,
        'timestamp': datetime.now().isoformat()
    }

def run_test_suite(model_name, suite_name='complete', output_dir=None, verbose=False):
    """Run a complete test suite"""
    
    # Get the test suite
    if suite_name not in TEST_SUITES:
        raise ValueError(f"Unknown test suite: {suite_name}")
    
    suite = TEST_SUITES[suite_name]
    
    print(f"\n{'='*60}")
    print(f"Running Test Suite: {suite_name}")
    print(f"Description: {suite['description']}")
    print(f"Model: {model_name}")
    print(f"Number of tests: {len(suite['tests'])}")
    print('='*60)
    
    # Initialize model interface
    model = get_model_interface(model_name)
    
    # Run all tests
    results = []
    for test_name, test_func in suite['tests']:
        result = run_single_test(model, test_name, test_func, verbose)
        results.append(result)
    
    # Aggregate results
    summary = aggregate_results(results)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print('='*60)
    print(f"Total tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Partial: {summary['partial']} ⚠️")
    print(f"Overall score: {summary['overall_score']:.1%}")
    
    # Category breakdown
    print("\nBy Category:")
    for category, score in summary['by_category'].items():
        print(f"  {category}: {score:.1%}")
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = output_path / f"results_{model_name.replace('/', '_')}_{suite_name}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'model': model_name,
                'suite': suite_name,
                'summary': summary,
                'detailed_results': results
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
        
        # Save summary
        summary_file = output_path / f"summary_{model_name.replace('/', '_')}_{suite_name}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Suite: {suite_name}\n")
            f.write(f"Date: {datetime.now().isoformat()}\n")
            f.write(f"\nOverall Score: {summary['overall_score']:.1%}\n")
            f.write(f"Passed: {summary['passed']}/{summary['total_tests']}\n")
            f.write("\nDetailed Results:\n")
            for r in results:
                status = '✅' if r['evaluation']['passed'] else '❌'
                f.write(f"  {r['test_name']}: {status} ({r['evaluation']['score']:.2f})\n")
        
        print(f"Summary saved to: {summary_file}")
    
    return summary

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run minAction.net LLM physics understanding tests'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model to test (e.g., qwen2-math:7b, gpt-4, claude-3)'
    )
    
    parser.add_argument(
        '--suite',
        type=str,
        default='complete',
        choices=list(TEST_SUITES.keys()),
        help='Test suite to run'
    )
    
    parser.add_argument(
        '--test',
        type=str,
        help='Run a single test by name'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--list-tests',
        action='store_true',
        help='List available tests and exit'
    )
    
    args = parser.parse_args()
    
    # List tests if requested
    if args.list_tests:
        print("\nAvailable Test Suites:")
        for name, suite in TEST_SUITES.items():
            print(f"\n{name}: {suite['description']}")
            print("  Tests:")
            for test_name, _ in suite['tests']:
                print(f"    - {test_name}")
        return
    
    # Run single test if specified
    if args.test:
        # Find the test
        test_func = None
        for suite in TEST_SUITES.values():
            for name, func in suite['tests']:
                if name == args.test:
                    test_func = func
                    break
        
        if not test_func:
            print(f"Error: Unknown test '{args.test}'")
            return 1
        
        # Run the test
        model = get_model_interface(args.model)
        result = run_single_test(model, args.test, test_func, args.verbose)
        
        # Print result
        status = '✅ PASSED' if result['evaluation']['passed'] else '❌ FAILED'
        print(f"\n{status}")
        print(f"Score: {result['evaluation']['score']:.2f}")
        
    else:
        # Run test suite
        summary = run_test_suite(
            args.model,
            args.suite,
            args.output,
            args.verbose
        )
        
        # Return non-zero if overall score < 50%
        if summary['overall_score'] < 0.5:
            return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
