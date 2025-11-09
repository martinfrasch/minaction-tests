# Multi-Model Testing Guide

This guide explains how to use the expanded testing infrastructure to evaluate multiple sophisticated LLM architectures on the minAction.net hypothesis.

## Overview

The testing infrastructure now supports:
- **20+ pre-configured LLM architectures** across multiple providers
- **Batch testing** across multiple models simultaneously
- **Comparative analysis** with visualization tools
- **Flexible model configurations** via YAML

## Quick Start

### 1. Test a Single Model

```bash
# Test with OpenAI GPT-4o
python scripts/run_tests.py --model gpt-4o --suite complete --output results/gpt4o

# Test with Claude 3.5 Sonnet
python scripts/run_tests.py --model claude-3-5-sonnet-20241022 --suite complete

# Test with local Ollama model
python scripts/run_tests.py --model llama3.1:70b --suite complete
```

### 2. Batch Test Multiple Models

```bash
# Test frontier models (requires API keys)
python scripts/run_batch_tests.py --config frontier_comparison --suite complete

# Test open-source models (local via Ollama)
python scripts/run_batch_tests.py --config open_source_comparison --suite complete

# Test specific models
python scripts/run_batch_tests.py --models gpt-4o claude-3-5-sonnet-20241022 llama3.1:70b
```

### 3. Compare Results

```bash
# Print comparison table
python scripts/compare_models.py --input results/batch --detailed

# Export to CSV
python scripts/compare_models.py --input results/batch --export comparison.csv

# Generate visualizations
python scripts/compare_models.py --input results/batch --visualize --output-dir results/viz
```

## Available Models

### Proprietary Models (API Required)

#### OpenAI
- `gpt-4o` - Latest flagship model
- `gpt-4o-mini` - Efficient variant
- `gpt-4-turbo` - Enhanced GPT-4
- `gpt-4` - Original GPT-4
- `gpt-3.5-turbo` - Previous generation

**Setup**: Set `OPENAI_API_KEY` environment variable

#### Anthropic Claude
- `claude-3-5-sonnet-20241022` - Latest Claude (recommended)
- `claude-3-opus-20240229` - Most capable Claude 3
- `claude-3-sonnet-20240229` - Balanced performance
- `claude-3-haiku-20240307` - Fast variant

**Setup**: Set `ANTHROPIC_API_KEY` environment variable

### Open-Source Models (via Ollama)

#### Math-Specialized
- `qwen2-math:7b` - Qwen mathematical reasoning (baseline)
- `qwen2.5:72b` - Latest Qwen with enhanced reasoning
- `deepseek-math:7b` - DeepSeek mathematical model

#### General Purpose
- `llama3.1:70b` - Meta's flagship (70B parameters)
- `llama3.1:8b` - Efficient Llama variant
- `llama3.2:3b` - Smallest Llama 3
- `phi3:14b` - Microsoft's compact model

#### Advanced Architectures
- `mistral-large:latest` - Mistral's largest (123B)
- `mixtral:8x7b` - Mixture of Experts (47B effective)

**Setup**: Install Ollama (`https://ollama.ai`) - models auto-download on first use

### HuggingFace Models (Direct Inference)

```python
# Requires GPU and transformers library
python scripts/run_tests.py --model deepseek-ai/deepseek-math-7b-instruct
python scripts/run_tests.py --model Qwen/Qwen2-Math-72B-Instruct
```

**Setup**: Requires CUDA-capable GPU for larger models

## Test Configurations

Pre-configured test suites in `config/model_configs.yaml`:

### `quick_comparison`
Fast test with diverse architectures (4 models)
- GPT-4o Mini
- Claude 3 Haiku
- Qwen2-Math 7B
- Llama 3.1 8B

### `frontier_comparison`
Flagship models from major providers (4 models)
- GPT-4o
- Claude 3.5 Sonnet
- Llama 3.1 70B
- Mistral Large

### `open_source_comparison`
Comprehensive open-source evaluation (5 models)
- Qwen2-Math 7B
- Llama 3.1 70B
- Mixtral 8x7B
- DeepSeek-Math 7B
- Phi-3 14B

### `math_specialized`
Math-focused models (4 models)
- Qwen2-Math 7B
- Qwen2.5 72B
- DeepSeek-Math 7B
- GPT-4o

### `size_scaling`
Test effect of model size (5 models)
- Llama 3.2 3B
- Llama 3.1 8B
- Llama 3.1 70B
- Qwen2-Math 7B
- Qwen2.5 72B

## Test Suites

### `complete` (9 tests - recommended)
All tests from the paper:
- Forward derivation (4 tests)
- Inverse problems (1 test)
- Physical constraints (2 tests)
- Cross-domain (1 test)
- Symmetry (1 test)

### `forward_only` (4 tests)
Tests mechanical equation manipulation ability

### `understanding_only` (3 tests)
Tests physical intuition and constraint understanding

## Usage Examples

### Example 1: Test Frontier Models

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Run batch test
python scripts/run_batch_tests.py \
  --config frontier_comparison \
  --suite complete \
  --output results/frontier \
  --verbose

# Compare results
python scripts/compare_models.py \
  --input results/frontier \
  --export results/frontier/comparison.csv \
  --visualize \
  --detailed
```

### Example 2: Local Open-Source Testing

```bash
# Install Ollama first: https://ollama.ai

# Run batch test (models auto-download)
python scripts/run_batch_tests.py \
  --config open_source_comparison \
  --suite complete \
  --output results/open_source

# Generate comparison report
python scripts/compare_models.py \
  --input results/open_source \
  --export results/open_source_report.md \
  --visualize
```

### Example 3: Custom Model Selection

```bash
# Test specific models of interest
python scripts/run_batch_tests.py \
  --models gpt-4o gpt-4-turbo gpt-3.5-turbo \
  --suite understanding_only \
  --output results/gpt_comparison

# Compare
python scripts/compare_models.py --input results/gpt_comparison
```

### Example 4: Size Scaling Study

```bash
# Test how model size affects understanding
python scripts/run_batch_tests.py \
  --config size_scaling \
  --suite complete \
  --output results/size_scaling

# Analyze results
python scripts/compare_models.py \
  --input results/size_scaling \
  --export results/size_analysis.csv \
  --detailed
```

## Output Format

### Individual Model Results

```
results/batch/
├── gpt-4o_complete.json          # Full results with prompts/responses
├── claude-3-5-sonnet_complete.json
└── llama3.1_70b_complete.json
```

Each file contains:
- Model name and configuration
- Summary scores (overall, by category)
- Detailed test results with prompts and responses
- Evaluation criteria and scoring

### Batch Summary

```json
{
  "test_suite": "complete",
  "total_models": 4,
  "successful_models": 4,
  "models_tested": ["gpt-4o", "claude-3-5-sonnet-20241022", ...],
  "results": [...]
}
```

### Comparison Output

#### CSV Format
```csv
Model,Overall Score,Passed,Failed,forward_derivation,inverse_problems,...
gpt-4o,0.85,7,2,0.95,0.60,...
claude-3-5-sonnet,0.82,7,2,0.92,0.55,...
```

#### Markdown Format
```markdown
# Model Comparison Results

## Summary
| Model | Overall Score | Passed | Failed |
|-------|---------------|--------|--------|
| gpt-4o | 85.0% | 7 | 2 |
...
```

### Visualizations

Generated PNG files:
- `overall_scores.png` - Bar chart of model performance
- `category_heatmap.png` - Performance by test category
- `detailed_heatmap.png` - Individual test scores

## Adding New Models

### Via YAML Configuration

Edit `config/model_configs.yaml`:

```yaml
models:
  ollama:
    - name: "new-model:latest"
      display_name: "New Model"
      description: "Description"
      size: "7B"
      category: "open-source"
      tested: false
```

### Via Command Line

Just specify the model name:

```bash
python scripts/run_tests.py --model your-new-model:tag
```

The system auto-detects the provider:
- Contains `:` → Ollama
- Starts with `gpt` → OpenAI
- Starts with `claude` → Anthropic
- Otherwise → HuggingFace

## Interpreting Results

### Overall Score
Percentage of tests passed (threshold: 70%)
- **0-30%**: Poor understanding
- **30-60%**: Partial mechanical ability
- **60-80%**: Good mechanical, limited understanding
- **80-100%**: Strong performance

### Category Scores

**Forward Derivation** (mechanical ability)
- Tests equation manipulation
- High scores expected for competent models

**Inverse Problems** (understanding)
- Tests reconstruction of Lagrangians
- Low scores indicate lack of physical intuition

**Physical Constraints** (deep understanding)
- Tests recognition of valid structures
- Critical for evaluating true understanding

### Key Insight from Paper

Models typically show:
- **87.5% on forward derivation** (can manipulate equations)
- **0% on inverse problems** (cannot understand structure)
- **Overall ~61%** (mechanical + understanding gap)

## Performance Optimization

### For API Models
- Use `--suite forward_only` for cheaper initial testing
- Set `--continue-on-error` to avoid stopping on failures
- Use `gpt-4o-mini` or `claude-3-haiku` for cost-effective testing

### For Local Models
- Ensure sufficient RAM (70B models need ~40GB)
- Use GPU for faster inference
- Test smaller models first (`llama3.1:8b` before `:70b`)

### For Batch Testing
- Run overnight for large model sets
- Use `--verbose` only for debugging
- Results auto-save after each model

## Troubleshooting

### API Key Errors
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Ollama Connection Errors
```bash
# Start Ollama service
ollama serve

# Verify models
ollama list
```

### Out of Memory
- Use smaller model variants (`:8b` instead of `:70b`)
- Close other applications
- Consider API-based models instead

### Missing Dependencies
```bash
pip install -r requirements.txt
pip install pyyaml  # If not installed
```

## Research Applications

### 1. Architecture Comparison Study
Compare how different architectures (transformer variants, MoE, etc.) handle physical reasoning

### 2. Scale Analysis
Investigate if model size correlates with physical understanding

### 3. Specialization Analysis
Test if math-specialized models outperform general models

### 4. Provider Comparison
Compare OpenAI vs Anthropic vs open-source on understanding tasks

### 5. Hypothesis Validation
Replicate paper findings across modern architectures

## Citation

If you use this testing infrastructure in research, please cite:

```bibtex
@article{minaction2024,
  title={minAction.net: Training LLMs on Physical Selection Principles},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  doi={10.5281/zenodo.14194603}
}
```

## Support

- Issues: https://github.com/[repo]/issues
- Documentation: See `README.md` for project overview
- Configuration: `config/model_configs.yaml`
