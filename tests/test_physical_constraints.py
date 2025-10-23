"""
Test Category 3: Physical Constraints
Tests 6-7: Recognize invalid Lagrangians and fundamental constraints
"""

def test_invalid_lagrangian(model):
    """
    Test 6: Identify problems with L = ẋ³ - x²
    Should recognize violation of classical mechanics principles
    """
    prompt = """A student proposes: L = ẋ³ - x²

What's wrong with this Lagrangian for classical mechanics?
What physical principles does it violate?

Consider:
1. What form should kinetic energy take in classical mechanics?
2. What does the momentum p = ∂L/∂ẋ become with this Lagrangian?
3. Is the resulting physics sensible?"""
    
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_6_invalid_lagrangian',
        'category': 'physical_constraints',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['quadratic', 'ẋ²', 'not cubic', 'violates', 'classical mechanics'],
        'validation_criteria': 'recognizes_violation'
    }

def test_higher_derivatives(model):
    """
    Test 7: Explain why L must be L(x,ẋ) not L(x,ẋ,ẍ)
    Tests understanding of fundamental constraints
    """
    prompt = """Why must a classical mechanics Lagrangian be L(x,ẋ) and not L(x,ẋ,ẍ)? 
What principle does this preserve?

Explain the physical and mathematical reasons for this constraint."""
    
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_7_higher_derivatives',
        'category': 'physical_constraints',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['Ostrogradsky', 'instability', 'determinism', 'initial conditions', 'bounded'],
        'validation_criteria': 'understands_constraints'
    }
