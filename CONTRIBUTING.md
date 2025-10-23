# Contributing to minAction-LLM-Physics-Tests

We welcome contributions that help advance our understanding of how AI systems learn physical principles!

## How to Contribute

### 1. Testing Additional Models

We encourage testing more models, especially:
- Physics-trained models (interesting comparison)
- Larger mathematical models
- Multi-modal models that can see equations
- Your own fine-tuned models

To add results for a new model:
1. Fork the repository
2. Run the test suite: `python scripts/run_tests.py --model your-model`
3. Save results in `results/your-model/`
4. Submit a pull request with your findings

### 2. Adding New Test Cases

Good test cases:
- Test understanding, not memorization
- Have clear expected outcomes
- Reveal something about physical intuition
- Can be evaluated objectively

To add a test:
1. Create test function in appropriate module
2. Add evaluation criteria to `src/evaluation.py`
3. Document expected behavior
4. Include in test suite

Example structure:
```python
def test_conservation_of_energy(model):
    """
    Test if model understands energy conservation from Lagrangian
    """
    prompt = "Your clear, specific prompt here"
    response = model.generate(prompt)
    
    return {
        'test_id': 'test_energy_conservation',
        'category': 'conservation_laws',
        'prompt': prompt,
        'response': response,
        'expected_elements': ['conserved', 'energy', 'time-invariant'],
        'validation_criteria': 'understands_energy_conservation'
    }
```

### 3. Improving Evaluation

Current evaluation is rule-based. We welcome:
- More sophisticated scoring methods
- Semantic similarity measures
- Automated proof checking
- Statistical analysis of results

### 4. Extending to New Domains

Beyond physics, test understanding of:
- Chemical reaction dynamics
- Ecological systems
- Economic equilibria
- Neural dynamics

Key: Focus on whether models understand selection principles, not just pattern matching.

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/minAction-LLM-physics-tests.git
cd minAction-LLM-physics-tests

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install -e .

# Run tests
pytest tests/
```

## Testing Guidelines

Before submitting:
1. Ensure all tests pass: `pytest`
2. Check code style: `black .` and `flake8`
3. Update documentation if needed
4. Add your results to the results table

## Pull Request Process

1. Fork and create feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit with clear messages
3. Push to your fork: `git push origin feature/your-feature`
4. Submit PR with:
   - Clear description of changes
   - Test results if applicable
   - Any new dependencies

## Code Style

- Python: Follow PEP 8
- Use type hints where helpful
- Document functions with docstrings
- Keep prompts readable and well-formatted

## Reporting Issues

When reporting issues, include:
- Model tested
- Full error message
- System details (OS, Python version)
- Steps to reproduce

## Questions?

Open an issue for:
- Clarification on test methodology
- Suggestions for new test categories
- Discussion of results interpretation

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Any resulting publications
- Conference presentations

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Contact

- GitHub Issues: Best for bugs and features
- Email: mfrasch@uw.edu for research collaborations
