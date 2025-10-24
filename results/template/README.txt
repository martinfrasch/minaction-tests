This is a template directory for storing your test results.

When you run tests, results will be saved in this format:

results/
  └── your_model_name/
      ├── summary.txt              # Human-readable summary
      ├── detailed_results.json    # Complete results with metadata
      └── figures/                 # Generated visualizations
          ├── category_breakdown.png
          ├── test_heatmap.png
          └── phase_comparison.png

To run tests and save results here:

python scripts/run_tests.py \
    --model your-model:version \
    --output results/your_model_name/

To analyze and visualize:

python scripts/analyze_results.py \
    --input results/your_model_name/ \
    --output results/your_model_name/figures/
