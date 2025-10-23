"""
Test Category 1: Forward Derivation
Tests 1-4: Apply Euler-Lagrange equation to derive equations from Lagrangians
"""

def test_general_lagrangian(model):
    """
    Test 1: General Lagrangian L = ½m*ẋ² - V(x)
    Should derive F = ma form
    """
    prompt = """Given the Lagrangian L = (1/2)m*ẋ² - V(x), apply the Euler-Lagrange equation d/dt(∂L/∂ẋ) - ∂L/∂x = 0 to derive the equation of motion. Show all steps.

This is a fundamental test of variational calculus applied to physics."""
    
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_1_general_lagrangian',
        'category': 'forward_derivation',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['mẍ', 'dV/dx', 'F = ma'],
        'validation_criteria': 'correct_euler_lagrange_general'
    }

def test_harmonic_oscillator(model):
    """
    Test 2: Harmonic oscillator L = ½m*ẋ² - ½k*x²
    Should derive mẍ + kx = 0
    """
    prompt = """For a harmonic oscillator with Lagrangian L = (1/2)m*ẋ² - (1/2)k*x², use the Euler-Lagrange equation to derive the equation of motion. Show: 1) Compute ∂L/∂x, 2) Compute ∂L/∂ẋ, 3) Apply d/dt to ∂L/∂ẋ, 4) Write final equation."""
    
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_2_harmonic_oscillator',
        'category': 'forward_derivation',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['mẍ', 'kx', 'mẍ + kx = 0'],
        'validation_criteria': 'correct_harmonic_oscillator'
    }

def test_novel_lagrangian_1(model):
    """
    Test 3: Novel Lagrangian with random coefficients
    L = 2.3*ẋ² - 1.5*x² - 0.8*x⁴
    Tests true understanding vs memorization
    """
    prompt = """Given a novel Lagrangian L = 2.3*ẋ² - 1.5*x² - 0.8*x⁴, derive the equation of motion using Euler-Lagrange equation. This Lagrangian has never appeared in training data.

Show all mathematical steps."""
    
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_3_novel_lagrangian_1',
        'category': 'forward_derivation',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['4.6*ẍ', '3.0*x', '3.2*x³'],
        'validation_criteria': 'correct_novel_1'
    }

def test_novel_lagrangian_2(model):
    """
    Test 4: Another novel Lagrangian
    L = 3.714*ẋ² - 2.156*x² - 0.823*x⁴
    """
    prompt = """Apply Euler-Lagrange equation EXACTLY as written: d/dt(∂L/∂ẋ) - ∂L/∂x = 0 to the Lagrangian L = 3.714*ẋ² - 2.156*x² - 0.823*x⁴. Show each step carefully with NO shortcuts."""
    
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_4_novel_lagrangian_2',
        'category': 'forward_derivation',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['7.428*ẍ', '4.312*x', '3.292*x³'],
        'validation_criteria': 'correct_novel_2'
    }
