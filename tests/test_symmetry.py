"""
Test Category 5: Symmetry and Conservation
Test 9: Connect symmetries to conservation laws via Noether's theorem
"""

def test_rotational_symmetry(model):
    """
    Test 9: Noether's theorem - rotational invariance
    Should identify angular momentum conservation
    """
    prompt = """If a Lagrangian is invariant under rotation, what quantity is conserved according to Noether's theorem?

Explain the connection between symmetry and conservation."""
    
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_9_rotational_symmetry',
        'category': 'symmetry',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['angular momentum', 'conserved', 'Noether'],
        'validation_criteria': 'correct_conservation_law'
    }
