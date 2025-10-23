"""
Test Category 2: Inverse Problems
Test 5: Find Lagrangians that produce given equations
"""

def test_find_lagrangian(model):
    """
    Test 5: Reverse engineering - find L from equation
    Given: ẍ + 3.1*x + 1.7*x³ = 0
    Should find: L = ½*ẋ² - (3.1/2)*x² - (1.7/4)*x⁴
    """
    prompt = """A system obeys the equation of motion: ẍ + 3.1*x + 1.7*x³ = 0

Find a Lagrangian L(x,ẋ) that produces this equation via the principle of least action.
Is your answer unique? Explain.

Note: The Lagrangian should be a function of position x and velocity ẋ only, not acceleration."""
    
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_5_find_lagrangian',
        'category': 'inverse_problems',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['L(x,ẋ)', 'not L(x,ẋ,ẍ)', 'ẋ²', 'x²', 'x⁴'],
        'validation_criteria': 'no_acceleration_terms'
    }
