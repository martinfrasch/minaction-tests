#!/bin/bash

# test_selection_principles.sh
# Tests whether adding explicit selection principles improves model performance

echo "=============================================="
echo "Testing Physics Discovery with Selection Principles"
echo "Model: qwen2-math:7b"
echo "Date: $(date)"
echo "=============================================="

# Create output directory
mkdir -p results/selection_principles
OUTPUT_FILE="results/selection_principles/results_$(date +%Y%m%d_%H%M%S).txt"

# Function to run test and save output
run_test() {
    local test_name=$1
    local prompt=$2
    
    echo ""
    echo "-------------------------------------------"
    echo "TEST: $test_name"
    echo "-------------------------------------------"
    echo ""
    
    # Save to file
    echo "" >> "$OUTPUT_FILE"
    echo "=== $test_name ===" >> "$OUTPUT_FILE"
    echo "PROMPT:" >> "$OUTPUT_FILE"
    echo "$prompt" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "RESPONSE:" >> "$OUTPUT_FILE"
    
    # Run ollama and capture output
    response=$(ollama run qwen2-math:7b "$prompt")
    echo "$response"
    echo "$response" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    
    # Brief pause to avoid overwhelming the model
    sleep 2
}

# Test 5 REVISED: Reverse Engineering with Selection Principles
run_test "Test 5 REVISED - Reverse Engineering" "Given the equation of motion: ẍ + 3.1*x + 1.7*x³ = 0

Find a Lagrangian using these PHYSICAL SELECTION PRINCIPLES:
1. Classical Lagrangians must be L(x,ẋ) only - never include ẍ
2. Kinetic energy T must be quadratic in velocities: T = ½m*ẋ²
3. Potential energy V(x) contains only position terms
4. The form is L = T - V

Apply these principles to construct L such that the Euler-Lagrange equation yields the given equation. Note: coefficients in your L will be half the coefficients in the equation for quadratic terms, and one-quarter for quartic terms."

# Test 6 REVISED: Invalid Lagrangian with Selection Criteria
run_test "Test 6 REVISED - Invalid Lagrangian" "A student proposes L = ẋ³ - x²

Evaluate this using PHYSICAL SELECTION CRITERIA:
1. Is kinetic energy quadratic in velocities? (Required for Galilean invariance)
2. Does this lead to F = ma form? (Required for Newtonian mechanics)
3. Is the momentum p = ∂L/∂ẋ linear in velocity? (Required for classical mechanics)
4. Would the Hamiltonian H = pẋ - L be bounded below? (Required for stability)

Apply these selection criteria to determine if this Lagrangian describes a valid classical system. If not, explain which principles it violates and suggest a correction."

# Test 7 REVISED: Why L(x,ẋ) with Selection Principles
run_test "Test 7 REVISED - Lagrangian Constraints" "Classical mechanics SELECTS Lagrangians of form L(x,ẋ) and REJECTS L(x,ẋ,ẍ).

Consider these SELECTION CRITERIA that nature applies:
1. Determinism: System state (x,ẋ) at t₀ must uniquely determine future
2. Energy stability: Hamiltonian must be bounded below (no infinite negative energy)
3. Ghost freedom: No spurious degrees of freedom (Ostrogradsky instability)
4. Causality: Solutions depend on initial conditions, not initial accelerations

Explain how L(x,ẋ,ẍ) violates these selection principles and why nature chooses L(x,ẋ) instead. What goes wrong physically if we allow ẍ in the Lagrangian?"

# Test 8 REVISED: Cross-Domain with Biological Selection
run_test "Test 8 REVISED - Population Dynamics" "Apply action principle to population dynamics with N(t) organisms.

Use these BIOLOGICAL SELECTION PRINCIPLES to choose your functional:
1. What quantity do organisms maximize over evolutionary time? (reproductive success, resource efficiency?)
2. What constraints exist? (carrying capacity K, limited resources, competition)
3. What is the 'kinetic' term? (rate of change, growth potential?)
4. What is the 'potential' term? (environmental resistance, crowding effects?)

Construct an action S[N(t)] that:
- Incorporates carrying capacity K
- Includes growth rate r
- Accounts for competition/crowding
- Yields realistic population dynamics (not just exponential growth)

Remember: The functional must make BIOLOGICAL sense, not just mathematical sense."

# Summary
echo ""
echo "=============================================="
echo "TESTING COMPLETE"
echo "=============================================="
echo "Results saved to: $OUTPUT_FILE"
echo ""
echo "Compare these results with the original failures to see if"
echo "explicit selection principles improve model performance."
echo "(Spoiler: They don't - the gap is understanding, not information)"
echo "=============================================="
