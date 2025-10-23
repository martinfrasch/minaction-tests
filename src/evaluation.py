"""
Evaluation module for scoring model responses
"""

import re
from typing import Dict, List, Any

def evaluate_response(test_result: Dict) -> Dict:
    """
    Evaluate a model's response to a test
    
    Returns:
        Dict with keys: passed, score, details
    """
    
    response = test_result['response'].lower()
    test_id = test_result['test_id']
    
    # Get validation criteria
    criteria = test_result.get('validation_criteria', 'general')
    expected = test_result.get('expected_elements', [])
    
    # Apply specific validation based on test type
    if criteria == 'correct_euler_lagrange_general':
        return _evaluate_general_lagrangian(response, expected)
    elif criteria == 'correct_harmonic_oscillator':
        return _evaluate_harmonic_oscillator(response, expected)
    elif criteria == 'correct_novel_1':
        return _evaluate_novel_1(response)
    elif criteria == 'correct_novel_2':
        return _evaluate_novel_2(response)
    elif criteria == 'no_acceleration_terms':
        return _evaluate_no_acceleration(response)
    elif criteria == 'recognizes_violation':
        return _evaluate_recognizes_violation(response)
    elif criteria == 'understands_constraints':
        return _evaluate_understands_constraints(response)
    elif criteria == 'biological_relevance':
        return _evaluate_biological_relevance(response)
    elif criteria == 'correct_conservation_law':
        return _evaluate_conservation_law(response)
    else:
        return _evaluate_general(response, expected)

def _evaluate_general_lagrangian(response: str, expected: List[str]) -> Dict:
    """Evaluate Test 1: General Lagrangian"""
    score = 0.0
    details = []
    
    # Check for key steps
    if 'd/dt' in response and '∂l/∂' in response:
        score += 0.3
        details.append("Shows differentiation steps")
    
    if 'mẍ' in response or 'm*a' in response or 'f = ma' in response.lower():
        score += 0.4
        details.append("Derives Newton's second law")
    
    if 'dv/dx' in response or '∂v/∂x' in response:
        score += 0.3
        details.append("Correctly handles potential")
    
    return {
        'passed': score >= 0.7,
        'score': score,
        'details': details
    }

def _evaluate_harmonic_oscillator(response: str, expected: List[str]) -> Dict:
    """Evaluate Test 2: Harmonic Oscillator"""
    score = 0.0
    details = []
    
    # Check for correct equation
    if 'mẍ + kx = 0' in response.replace(' ', '') or 'mẍ = -kx' in response.replace(' ', ''):
        score += 0.5
        details.append("Correct final equation")
    
    # Check for intermediate steps
    if '∂l/∂x' in response and '-kx' in response:
        score += 0.25
        details.append("Correct ∂L/∂x")
    
    if '∂l/∂ẋ' in response and 'mẋ' in response:
        score += 0.25
        details.append("Correct ∂L/∂ẋ")
    
    # Partial credit for errors in process but correct result
    if score >= 0.5 and len(details) < 3:
        return {
            'passed': False,
            'score': 0.5,
            'details': details + ["Errors in derivation but correct final answer"]
        }
    
    return {
        'passed': score >= 0.7,
        'score': score,
        'details': details
    }

def _evaluate_novel_1(response: str) -> Dict:
    """Evaluate Test 3: Novel Lagrangian 1"""
    score = 0.0
    details = []
    
    # Check for correct coefficients
    if '4.6' in response:
        score += 0.33
        details.append("Correct acceleration coefficient")
    
    if '3.0' in response or '3*x' in response:
        score += 0.33
        details.append("Correct linear term")
    
    if '3.2' in response:
        score += 0.34
        details.append("Correct cubic term")
    
    return {
        'passed': score >= 0.9,
        'score': score,
        'details': details
    }

def _evaluate_novel_2(response: str) -> Dict:
    """Evaluate Test 4: Novel Lagrangian 2"""
    score = 0.0
    details = []
    
    # Check for correct coefficients
    if '7.428' in response:
        score += 0.33
        details.append("Correct acceleration coefficient")
    
    if '4.312' in response:
        score += 0.33
        details.append("Correct linear term")
    
    if '3.292' in response:
        score += 0.34
        details.append("Correct cubic term")
    
    return {
        'passed': score >= 0.9,
        'score': score,
        'details': details
    }

def _evaluate_no_acceleration(response: str) -> Dict:
    """Evaluate Test 5: No acceleration in Lagrangian"""
    score = 0.0
    details = []
    
    # Critical: Check for invalid acceleration terms
    if 'ẍ' in response and 'l(' in response and 'ẍ' in response[response.find('l('):]:
        score = 0.0
        details.append("ERROR: Included acceleration in Lagrangian")
        return {
            'passed': False,
            'score': 0.0,
            'details': details
        }
    
    # Check for correct structure
    if 'l(x,ẋ)' in response.replace(' ', '') or 'l = ' in response:
        score += 0.5
        details.append("Lagrangian has correct arguments")
    
    if 'ẋ²' in response or 'kinetic' in response:
        score += 0.25
        details.append("Includes kinetic term")
    
    if 'x²' in response and 'x⁴' in response:
        score += 0.25
        details.append("Includes potential terms")
    
    return {
        'passed': score >= 0.7,
        'score': score,
        'details': details
    }

def _evaluate_recognizes_violation(response: str) -> Dict:
    """Evaluate Test 6: Recognizes invalid Lagrangian"""
    score = 0.0
    details = []
    
    # Check if recognizes the problem
    if 'quadratic' in response and 'velocity' in response:
        score += 0.5
        details.append("Identifies kinetic energy must be quadratic")
    
    if 'violate' in response or 'invalid' in response or 'wrong' in response:
        score += 0.25
        details.append("Recognizes violation")
    
    if 'classical mechanics' in response:
        score += 0.25
        details.append("References classical mechanics principles")
    
    return {
        'passed': score >= 0.7,
        'score': score,
        'details': details
    }

def _evaluate_understands_constraints(response: str) -> Dict:
    """Evaluate Test 7: Understands why no higher derivatives"""
    score = 0.0
    details = []
    
    keywords = ['ostrogradsky', 'instability', 'determinism', 'initial conditions', 
                'bounded', 'ghost', 'unbounded', 'energy']
    
    for keyword in keywords:
        if keyword in response:
            score += 0.2
            details.append(f"Mentions {keyword}")
    
    # Cap at 1.0
    score = min(score, 1.0)
    
    # Partial credit for mentioning principle of least action
    if score < 0.5 and 'least action' in response:
        score = 0.5
        details.append("Mentions principle of least action")
    
    return {
        'passed': score >= 0.7,
        'score': score,
        'details': details
    }

def _evaluate_biological_relevance(response: str) -> Dict:
    """Evaluate Test 8: Biological application"""
    score = 0.0
    details = []
    
    biological_terms = ['carrying capacity', 'growth', 'population', 'resources', 
                       'competition', 'predation', 'birth', 'death']
    
    terms_found = 0
    for term in biological_terms:
        if term in response:
            terms_found += 1
    
    if terms_found >= 3:
        score += 0.5
        details.append(f"Includes {terms_found} biological concepts")
    
    # Check if mechanically applies physics
    if 'l = n' in response.replace(' ', '') and 'dn/dt' in response:
        score = 0.5  # Partial credit for mechanical application
        details.append("Mechanical application without biological insight")
    
    return {
        'passed': score >= 0.7,
        'score': score,
        'details': details
    }

def _evaluate_conservation_law(response: str) -> Dict:
    """Evaluate Test 9: Conservation from symmetry"""
    score = 0.0
    details = []
    
    if 'angular momentum' in response:
        score += 0.7
        details.append("Correctly identifies angular momentum")
    
    if 'noether' in response:
        score += 0.15
        details.append("References Noether's theorem")
    
    if 'conserved' in response or 'conservation' in response:
        score += 0.15
        details.append("Discusses conservation")
    
    return {
        'passed': score >= 0.7,
        'score': score,
        'details': details
    }

def _evaluate_general(response: str, expected: List[str]) -> Dict:
    """General evaluation based on expected elements"""
    found = 0
    details = []
    
    for element in expected:
        if element.lower() in response:
            found += 1
            details.append(f"Found: {element}")
    
    score = found / len(expected) if expected else 0.0
    
    return {
        'passed': score >= 0.7,
        'score': score,
        'details': details
    }

def aggregate_results(results: List[Dict]) -> Dict:
    """
    Aggregate test results into summary statistics
    """
    total = len(results)
    passed = sum(1 for r in results if r['evaluation']['passed'])
    failed = total - passed
    
    # Calculate partial passes (score between 0.3 and 0.7)
    partial = sum(1 for r in results 
                  if 0.3 <= r['evaluation']['score'] < 0.7)
    
    # Category breakdown
    categories = {}
    for r in results:
        cat = r['result'].get('category', 'unknown')
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r['evaluation']['score'])
    
    category_scores = {
        cat: sum(scores) / len(scores) 
        for cat, scores in categories.items()
    }
    
    # Overall score
    overall = sum(r['evaluation']['score'] for r in results) / total if total > 0 else 0
    
    return {
        'total_tests': total,
        'passed': passed,
        'failed': failed - partial,
        'partial': partial,
        'overall_score': overall,
        'by_category': category_scores
    }
