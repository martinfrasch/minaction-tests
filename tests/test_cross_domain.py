"""
Test Category 4: Cross-Domain Transfer
Test 8: Apply variational principles beyond physics
"""

def test_population_dynamics(model):
    """
    Test 8: Apply action principle to population dynamics
    Tests understanding of variational principles in biological context
    """
    prompt = """Apply action principle to population dynamics with N(t) organisms. 
What functional should be minimized?

Consider:
1. What plays the role of 'kinetic energy' in population dynamics?
2. What plays the role of 'potential energy'?
3. What constraints exist (carrying capacity, resources)?
4. Propose a biologically meaningful action functional."""
    
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_8_population_dynamics',
        'category': 'cross_domain',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['carrying capacity', 'growth rate', 'resources', 'competition'],
        'validation_criteria': 'biological_relevance'
    }
