# Network-Weighted Gauge Theory: An AI-Assisted Exploration

## 🚨 Important Technical Note

**This repository contains exploratory computational work that requires mathematical reformulation.** During peer review, experts correctly identified that our discretization approach is not valid for arbitrary graphs. We're actively seeking collaborators with expertise in:

- **Discrete exterior calculus** on graphs
- **Cochain complexes** for gauge theory
- **Lattice gauge theory** on irregular graphs  
- **Graph homology** and cycle bases

If you have this expertise and find the patterns interesting, please contact: mfrasch@uw.edu

---

## Overview

This repository documents an AI-assisted exploration of gauge theory on network topologies, conducted through human-AI collaboration with Claude (Anthropic). While our initial mathematical framework requires correction, the work demonstrates:

1. **Novel AI collaboration methodology** for theoretical physics
2. **Interesting computational patterns** that may survive proper formalization
3. **Both successes and pitfalls** of AI-assisted research

## Key Findings (Requiring Validation)

Our computational experiments revealed intriguing patterns:
- Non-monotonic coupling dependence in three regimes
- Hierarchy between topological and local observables
- Topology-dependent scaling relationships

**However**, these findings used incorrect discretization (see Technical Issues below).

## Known Technical Issues 🔴

### 1. Invalid Discretization for Graphs
```python
# CURRENT (INCORRECT for arbitrary graphs):
A_transformed = A + np.gradient(Lambda)

# NEEDED (proper graph discretization):
A_transformed = A + incidence_matrix @ Lambda
```

### 2. Wilson Loop Invariance is Tautological
- Wilson loops are gauge-invariant by construction
- Their invariance doesn't demonstrate emergent symmetry
- Need to examine dynamical distributions instead

### 3. Missing Proper Gauge Structure
- Should use link variables U_ij ∈ U(1), not site variables
- Need gauge-covariant network penalties
- Requires proper cycle basis for non-planar graphs

### 4. Possible Optimization Artifacts
- "Phase transitions" lack finite-size scaling analysis
- Could be regularization effects, not genuine phases
- Need Binder cumulants and susceptibility peaks

## Repository Structure

```
minaction-tests/
├── code/
│   ├── current_implementation.py  # Original (flawed) implementation
│   ├── issues_identified.md       # Detailed technical problems
│   └── proper_discretization/     # Space for corrected version (help needed!)
├── results/
│   ├── computational_patterns/    # Interesting patterns found
│   └── validation_needed/         # Results requiring proper math
├── docs/
│   ├── ai_methodology.md         # How we used AI collaboration
│   └── physics_questions.md      # Open questions for collaborators
└── manuscripts/
    ├── cs_ai_version.tex         # AI methodology paper
    └── physics_version_draft.tex # Awaiting proper formulation
```

## How to Contribute

We explicitly invite collaboration to:

### For Physicists/Mathematicians:
1. **Implement proper discretization** using cochain complexes
2. **Define cycle bases** for Wilson loops on non-planar graphs  
3. **Add finite-size scaling** analysis
4. **Test if patterns persist** with correct mathematics

### For AI/ML Researchers:
1. **Extend AI collaboration methodology** to other domains
2. **Improve pattern recognition** approaches
3. **Develop validation frameworks** for AI-assisted theory

### Getting Started:
```bash
git clone https://github.com/martinfrasch/minaction-tests
cd minaction-tests
pip install -r requirements.txt

# Run original (flawed) implementation
python code/current_implementation.py

# See identified issues
cat code/issues_identified.md

# Help us implement proper version!
cd code/proper_discretization/
```

## Current Status

- ✅ AI methodology paper submitted to arXiv cs.AI
- ⚠️ Physics results need mathematical reformulation
- 🔄 Actively seeking collaborators
- 📖 Complete documentation of issues and patterns

## Why This Repository Matters

Even with mathematical flaws, this work demonstrates:
1. **How AI can accelerate theoretical exploration** (10-100x faster prototyping)
2. **The importance of domain expertise** (AI found patterns but also made errors)
3. **Value of transparency** in scientific process
4. **Potential for interdisciplinary collaboration**

## Papers

- **AI Methodology:** "AI-Assisted Theoretical Physics Exploration" (arXiv cs.AI, 2024)
- **Physics Paper:** In preparation, pending proper mathematical formulation

## The Patterns We Found

Despite mathematical issues, we observed:

### Non-Monotonic Coupling Dependence
- Weak regime (κ < 0.001): Perturbative behavior
- Intermediate (0.001 < κ < 0.5): Symmetry-breaking "desert"  
- Strong (κ > 0.5): Network-dominated regime

### Hierarchical Observable Behavior
- Wilson loops: ~10^-15 variation (but this is tautological)
- Local fields: ~30% variation (but using wrong discretization)

### Questions for Proper Investigation:
1. Do these regimes exist with proper cochain discretization?
2. Is non-monotonicity real or an artifact?
3. How do patterns change on different graph topologies?

## Contact & Collaboration

**Martin G. Frasch**  
Institute on Human Development and Disability  
University of Washington  
Email: mfrasch@uw.edu  
ORCID: 0000-0003-3159-6321

**Seeking experts in:**
- Discrete differential geometry
- Lattice gauge theory
- Graph theory and topology
- Critical phenomena and phase transitions

## Citation

If you use or build upon this work:

```bibtex
@article{frasch2024ai,
  title={AI-Assisted Theoretical Physics Exploration: A Case Study in Network-Weighted Gauge Theory},
  author={Frasch, Martin G.},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024},
  note={Manuscript acknowledges mathematical issues requiring reformulation}
}
```

## License

MIT License - Use freely but please acknowledge limitations

## Acknowledgments

- Claude (Anthropic) for AI collaboration
- Peer reviewers who identified critical mathematical issues
- Future collaborators who will help fix the mathematics

---

**Note:** This is living research. We've identified problems and are working to solve them. Science is a process, not just polished results. Join us in making this right! 🔬

# minAction-LLM-Physics-Tests

Testing whether mathematical language models understand physical selection principles through variational mechanics.

## Overview

This repository contains the empirical testing framework described in our paper "Variational Principles as Vertical Organizers: Testing Physical Understanding in Mathematical Language Models" (Frasch, 2025).

We tested whether current mathematical LLMs can:
1. Apply the Euler-Lagrange equation mechanically (✅ They can - 100% success)
2. Understand physical constraints on valid Lagrangians (❌ They cannot - 0% success)
3. Recognize selection principles even when explicitly provided (❌ They cannot - 0% improvement)

**Key Finding**: Current models achieve 61% success through mechanical application but lack understanding of how nature selects mathematical structures to become physical laws.

## Quick Start

```bash
# Clone repository
git clone https://github.com/martinfrasch/minAction-LLM-physics-tests.git
cd minAction-LLM-physics-tests

# Install dependencies
pip install -r requirements.txt

# Run basic test with Ollama
ollama pull qwen2-math:7b
python run_tests.py --model qwen2-math:7b --test forward_euler_lagrange

# Run full test suite
python run_tests.py --model qwen2-math:7b --suite complete
```

## Repository Structure

```
minAction-LLM-physics-tests/
├── README.md                          # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                            # MIT License
├── config/
│   └── model_configs.yaml            # 20+ LLM architecture configurations
├── docs/
│   └── MULTI_MODEL_TESTING.md        # Complete multi-model testing guide
├── paper/
│   ├── manuscript.md                  # Full paper with results
│   └── empirical_validation.md        # Detailed test results
├── tests/
│   ├── __init__.py
│   ├── test_forward_derivation.py     # Tests 1-4: Forward Euler-Lagrange
│   ├── test_inverse_problems.py       # Test 5: Reverse engineering
│   ├── test_physical_constraints.py   # Tests 6-7: Constraint recognition
│   ├── test_cross_domain.py          # Test 8: Population dynamics
│   └── test_symmetry.py              # Test 9: Noether's theorem
├── src/
│   ├── __init__.py
│   ├── model_interface.py            # Interface to LLMs (Ollama, HuggingFace, API)
│   ├── evaluation.py                 # Scoring and validation
│   ├── selection_principles.py       # Selection principle prompts
│   └── visualization.py              # Result plotting
├── prompts/
│   ├── phase1_basic/                 # Original test prompts
│   └── phase2_guided/                # With explicit selection principles
├── results/
│   ├── qwen2_math_7b/               # Our results
│   ├── batch/                        # Multi-model batch results
│   └── template/                     # Template for new model tests
├── notebooks/
│   ├── 01_reproduce_results.ipynb    # Reproduce our findings
│   ├── 02_test_new_model.ipynb      # Test your own model
│   └── 03_analyze_failures.ipynb     # Deep dive into failures
└── scripts/
    ├── run_tests.py                  # Main test runner
    ├── run_batch_tests.py            # Multi-model batch testing
    ├── compare_models.py             # Comparative analysis & visualization
    ├── analyze_results.py            # Generate statistics
    └── test_selection_principles.sh  # Bash script for phase 2 tests
```

## Test Categories

### Phase 1: Basic Understanding (9 tests)

| Category | Tests | What It Measures | Our Result |
|----------|-------|------------------|------------|
| Forward Derivation | 1-4 | Can apply Euler-Lagrange mechanically | 100% ✅ |
| Inverse Problems | 5 | Can find Lagrangians from equations | 0% ❌ |
| Physical Constraints | 6-7 | Recognizes invalid Lagrangians | 0% ❌ |
| Cross-Domain | 8 | Applies principles to biology | 50% ⚠️ |
| Symmetry | 9 | Understands Noether's theorem | 100% ✅ |

**Overall: 61% success rate**

### Phase 2: With Explicit Guidance (4 tests)

All failed tests were retried with explicit selection principles provided.
**Result: 0% improvement** - demonstrating the gap is understanding, not information.

## Running Tests

### Test a Single Problem

```python
from src.model_interface import OllamaInterface
from tests.test_forward_derivation import test_novel_lagrangian

model = OllamaInterface("qwen2-math:7b")
result = test_novel_lagrangian(model, coefficients=[2.3, 1.5, 0.8])
print(f"Success: {result['passed']}")
print(f"Model output: {result['response']}")
```

### Run Complete Test Suite

```python
from scripts.run_tests import run_complete_suite

results = run_complete_suite(
    model_name="qwen2-math:7b",
    output_dir="results/my_test"
)
print(f"Overall success rate: {results['overall_score']:.1%}")
```

### Test With Selection Principles

```bash
# This tests whether explicit guidance helps (it doesn't)
bash scripts/test_selection_principles.sh
```

## Key Findings

1. **Mechanical Proficiency**: Models can perfectly apply Euler-Lagrange to any Lagrangian
2. **Zero Physical Understanding**: Complete failure on recognizing physical constraints
3. **Guidance Doesn't Help**: Explicit principles don't improve performance
4. **Syntax Without Semantics**: Models learned mathematical procedures, not physical meaning

## Reproducing Our Results

### Prerequisites

- Python 3.8+
- Ollama or HuggingFace account
- 16GB RAM minimum (32GB recommended for larger models)
- ~10GB disk space for model weights

### Step-by-Step Reproduction

```bash
# 1. Install Ollama (if not installed)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull the model we tested
ollama pull qwen2-math:7b

# 3. Run our exact test suite
python scripts/run_tests.py --model qwen2-math:7b --suite exact_reproduction

# 4. Compare with our results
python scripts/analyze_results.py --compare results/qwen2_math_7b
```

Expected output:
```
Test 1 (General Lagrangian): ✅ PASS
Test 2 (Harmonic Oscillator): ⚠️ PARTIAL (errors but correct result)
Test 3 (Novel L=2.3ẋ²-1.5x²-0.8x⁴): ✅ PASS
...
Test 5 (Reverse Engineering): ❌ FAIL (included ẍ in Lagrangian)
...
Overall: 5.5/9 = 61%
```

## Testing Other Models

### Supported Interfaces

- **Ollama**: Local models (recommended for reproduction)
- **HuggingFace**: Direct transformer access
- **Google Gemini API**: Gemini models (1.5 Pro, Flash, etc.)
- **Anthropic API**: Claude models
- **Custom**: Implement `BaseModelInterface`

### Example: Testing Gemini

```python
from src.model_interface import GeminiInterface

# Set API key in environment
os.environ['GEMINI_API_KEY'] = 'your-key'

model = GeminiInterface("gemini-1.5-pro")
results = run_complete_suite(model, output_dir="results/gemini")
```

## Multi-Model Testing

**NEW**: The repository now supports batch testing across 20+ sophisticated LLM architectures!

### Quick Multi-Model Testing

```bash
# Test frontier models (Gemini 1.5 Pro, Claude 3.5 Sonnet, etc.)
export GEMINI_API_KEY="your-key"
export ANTHROPIC_API_KEY="sk-ant-..."
python scripts/run_batch_tests.py --config frontier_comparison

# Test open-source models (Llama 3.1, Mixtral, etc.)
python scripts/run_batch_tests.py --config open_source_comparison

# Compare results
python scripts/compare_models.py --input results/batch --visualize
```

### Available Model Architectures

**Proprietary Models**:
- Google Gemini: Gemini 2.0 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro
- Anthropic: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Sonnet, Claude 3 Haiku

**Open-Source Models** (via Ollama):
- Math-Specialized: Qwen2-Math 7B, Qwen2.5 72B, DeepSeek-Math 7B
- General: Llama 3.1 (70B/8B/3B), Phi-3 14B
- Advanced: Mistral Large 123B, Mixtral 8x7B (MoE)

### Pre-Configured Test Suites

- `quick_comparison`: Fast test with 4 diverse models
- `frontier_comparison`: Flagship models from major providers
- `open_source_comparison`: Comprehensive open-source evaluation
- `math_specialized`: Math-focused models
- `size_scaling`: Test effect of model size

### Complete Documentation

See [docs/MULTI_MODEL_TESTING.md](docs/MULTI_MODEL_TESTING.md) for:
- Detailed setup instructions
- Complete model list and configurations
- Batch testing workflows
- Comparative analysis guide
- Performance optimization tips
- Research applications

### Example: Compare Architecture Types

```bash
# Test different architectures
python scripts/run_batch_tests.py \
  --models gemini-1.5-pro claude-3-5-sonnet-20241022 llama3.1:70b mixtral:8x7b \
  --suite complete \
  --output results/arch_comparison

# Generate comparison report with visualizations
python scripts/compare_models.py \
  --input results/arch_comparison \
  --export results/comparison.csv \
  --visualize \
  --detailed
```

This produces:
- Comparative performance tables
- Category-wise analysis (forward vs. understanding tasks)
- Heat maps showing model strengths/weaknesses
- Statistical analysis across architectures

## Theoretical Background

The tests are based on fundamental physics principles:

1. **Euler-Lagrange Equation**: d/dt(∂L/∂ẋ) - ∂L/∂x = 0
2. **Physical Constraints**:
   - Kinetic energy must be quadratic in velocities
   - Lagrangians cannot depend on acceleration
   - Hamiltonian must be bounded below
3. **Selection Principles**: How nature chooses valid mathematical structures

See our [paper](paper/manuscript.md) for detailed theoretical framework.

## Contributing

We welcome contributions! Areas of interest:

1. **Test Additional Models**: Run batch tests on new architectures and contribute results
2. **Model Configurations**: Add new models to `config/model_configs.yaml`
3. **New Test Cases**: Particularly for field theories and quantum mechanics
4. **Biological Tests**: Extend cross-domain testing to other fields
5. **Visualization**: Better ways to show understanding gaps and comparative analysis
6. **Analysis Tools**: Enhanced statistical analysis for multi-model comparisons

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this testing framework, please cite:

```bibtex
@article{frasch2025variational,
  title={Variational Principles as Vertical Organizers: Testing Physical Understanding in Mathematical Language Models},
  author={Frasch, Martin G.},
  journal={zenodo [preprint] (https://doi.org/10.5281/zenodo.17437295)},
  year={2025}
}
```

 ```
[DOI: 10.5281/zenodo.17437295](https://doi.org/10.5281/zenodo.17437295)
```


## Related Work

- [Frasch et al. (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10375982/): Energy constraints in neurodevelopment
- [Frasch (2023)](https://arxiv.org/abs/2310.03042): Neural Architecture Search with biological principles
- [Frasch (2025)](https://philsci-archive.pitt.edu/26949/): Vertically organizing principles

## License

MIT License - See [LICENSE](LICENSE) file

## Contact

Martin G. Frasch
- Email: mfrasch@uw.edu
- Institution: University of Washington, Institute on Human Development and Disability
- GitHub: [@martinfrasch](https://github.com/martinfrasch)

## Acknowledgments

Thanks to the Qwen team for making their mathematical model available for testing.
