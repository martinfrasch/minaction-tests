"""
Selection principles for evaluating physical validity of mathematical structures.

This module implements the three core selection principles used in minAction.net:
1. Action minimization
2. Gauge invariance
3. Renormalizability
"""

import numpy as np
import sympy as sp
from typing import Dict, Tuple, List, Optional


class SelectionPrinciples:
    """
    Evaluator for physical selection principles.
    
    These principles determine which mathematical structures from the Platonic
    world (M) are physically realizable (P).
    """
    
    def __init__(self):
        """Initialize selection principle evaluators."""
        self.action_threshold = 0.1
        self.gauge_tolerance = 1e-6
        self.max_dimension = 4  # In 4D spacetime for renormalizability
    
    def evaluate_action_principle(self, 
                                  lagrangian: callable,
                                  path: np.ndarray,
                                  t_span: Tuple[float, float]) -> Dict:
        """
        Evaluate whether a Lagrangian satisfies action minimization.
        
        Tests whether the given path extremizes the action functional:
        S = ∫L(q, q̇, t) dt
        
        Args:
            lagrangian: Function L(q, q_dot, t) → scalar
            path: Array of shape (n_points, n_dof) representing a trajectory
            t_span: (t_start, t_end) time interval
            
        Returns:
            Dictionary with:
                - action: Value of action along path
                - is_extremum: Boolean, whether action is extremal
                - perturbation_actions: Actions for perturbed paths
                - percentile: Percentile rank (0-100) of original action
        """
        t = np.linspace(t_span[0], t_span[1], len(path))
        dt = t[1] - t[0]
        
        # Compute velocity by finite differences
        q_dot = np.gradient(path, dt, axis=0)
        
        # Compute action along original path
        action = 0.0
        for i in range(len(t)):
            action += lagrangian(path[i], q_dot[i], t[i]) * dt
        
        # Generate random perturbations
        n_perturbations = 1000
        perturbation_actions = []
        
        for _ in range(n_perturbations):
            # Add small random perturbation
            noise = np.random.randn(*path.shape) * 0.1
            perturbed_path = path + noise
            perturbed_q_dot = np.gradient(perturbed_path, dt, axis=0)
            
            # Compute action for perturbed path
            perturbed_action = 0.0
            for i in range(len(t)):
                perturbed_action += lagrangian(
                    perturbed_path[i], 
                    perturbed_q_dot[i], 
                    t[i]
                ) * dt
            
            perturbation_actions.append(perturbed_action)
        
        # Check if original action is extremal
        perturbation_actions = np.array(perturbation_actions)
        percentile = np.sum(perturbation_actions > action) / len(perturbation_actions) * 100
        
        # Action is extremal if it's in lowest 5% or highest 95%
        is_extremum = (percentile < 5) or (percentile > 95)
        
        # For physical systems, we expect minimum (not maximum)
        is_minimum = percentile < 5
        
        return {
            'action': action,
            'is_extremum': is_extremum,
            'is_minimum': is_minimum,
            'perturbation_actions': perturbation_actions,
            'percentile': percentile,
            'mean_perturbation_action': np.mean(perturbation_actions),
            'std_perturbation_action': np.std(perturbation_actions)
        }
    
    def check_gauge_invariance(self,
                               lagrangian_expr: sp.Expr,
                               variables: List[sp.Symbol],
                               gauge_transformation: callable) -> Dict:
        """
        Check if Lagrangian is invariant under gauge transformation.
        
        A Lagrangian L is gauge invariant if:
        L(φ + δφ) = L(φ) + total derivative
        
        where δφ is a gauge transformation.
        
        Args:
            lagrangian_expr: Symbolic Lagrangian expression
            variables: List of field variables
            gauge_transformation: Function that returns gauge variation δφ
            
        Returns:
            Dictionary with:
                - is_gauge_invariant: Boolean
                - variation: Symbolic variation of L
                - total_derivative_part: Total derivative term
        """
        # Compute variation of Lagrangian
        delta_L = sp.sympify(0)
        
        for var in variables:
            delta_var = gauge_transformation(var)
            delta_L += sp.diff(lagrangian_expr, var) * delta_var
        
        # Check if variation is a total derivative
        # (Simplified check: variation should be small)
        is_invariant = delta_L.simplify() == 0
        
        return {
            'is_gauge_invariant': is_invariant,
            'variation': delta_L,
            'simplified_variation': delta_L.simplify()
        }
    
    def check_renormalizability(self,
                                lagrangian_expr: sp.Expr,
                                space_dim: int = 4) -> Dict:
        """
        Check if Lagrangian is renormalizable.
        
        In d-dimensional spacetime, operators in the Lagrangian must have
        mass dimension ≤ d for the theory to be renormalizable.
        
        For standard physics: d = 4 (3 space + 1 time)
        
        Args:
            lagrangian_expr: Symbolic Lagrangian expression
            space_dim: Spacetime dimension (default 4)
            
        Returns:
            Dictionary with:
                - is_renormalizable: Boolean
                - operator_dimensions: Dict mapping terms to their mass dimensions
                - violating_terms: List of terms with dimension > space_dim
        """
        # Extract all terms in the Lagrangian
        terms = lagrangian_expr.as_ordered_terms()
        
        operator_dimensions = {}
        violating_terms = []
        
        for term in terms:
            # Estimate mass dimension (simplified)
            # Actual implementation would need proper dimensional analysis
            
            # Count derivatives (each adds dimension 1)
            n_derivatives = len([op for op in sp.preorder_traversal(term) 
                                if isinstance(op, sp.Derivative)])
            
            # Count field factors (each has specific dimension)
            # This is simplified - real implementation needs field dimension table
            n_fields = len(term.free_symbols)
            
            # Estimate dimension
            estimated_dim = n_derivatives + n_fields
            
            operator_dimensions[str(term)] = estimated_dim
            
            if estimated_dim > space_dim:
                violating_terms.append({
                    'term': str(term),
                    'dimension': estimated_dim
                })
        
        is_renormalizable = len(violating_terms) == 0
        
        return {
            'is_renormalizable': is_renormalizable,
            'operator_dimensions': operator_dimensions,
            'violating_terms': violating_terms,
            'space_dim': space_dim
        }
    
    def evaluate_all(self,
                    lagrangian: callable,
                    lagrangian_expr: Optional[sp.Expr],
                    path: np.ndarray,
                    t_span: Tuple[float, float],
                    variables: Optional[List[sp.Symbol]] = None,
                    gauge_transform: Optional[callable] = None) -> Dict:
        """
        Evaluate all three selection principles for a given Lagrangian.
        
        Args:
            lagrangian: Numerical Lagrangian function
            lagrangian_expr: Symbolic Lagrangian expression
            path: Trajectory to evaluate
            t_span: Time interval
            variables: Symbolic variables (for gauge check)
            gauge_transform: Gauge transformation (optional)
            
        Returns:
            Dictionary with results from all three principle checks
        """
        results = {}
        
        # 1. Action principle
        results['action'] = self.evaluate_action_principle(
            lagrangian, path, t_span
        )
        
        # 2. Gauge invariance (if symbolic expression provided)
        if lagrangian_expr is not None and variables is not None:
            if gauge_transform is not None:
                results['gauge'] = self.check_gauge_invariance(
                    lagrangian_expr, variables, gauge_transform
                )
            else:
                results['gauge'] = {'status': 'No gauge transformation provided'}
        else:
            results['gauge'] = {'status': 'No symbolic expression provided'}
        
        # 3. Renormalizability (if symbolic expression provided)
        if lagrangian_expr is not None:
            results['renormalizability'] = self.check_renormalizability(
                lagrangian_expr
            )
        else:
            results['renormalizability'] = {'status': 'No symbolic expression provided'}
        
        # Overall assessment
        action_pass = results['action']['is_minimum']
        gauge_pass = results.get('gauge', {}).get('is_gauge_invariant', None)
        renorm_pass = results.get('renormalizability', {}).get('is_renormalizable', None)
        
        results['summary'] = {
            'action_minimization': 'PASS' if action_pass else 'FAIL',
            'gauge_invariance': 'PASS' if gauge_pass else ('FAIL' if gauge_pass is False else 'N/A'),
            'renormalizability': 'PASS' if renorm_pass else ('FAIL' if renorm_pass is False else 'N/A'),
            'overall_physical_validity': action_pass and (gauge_pass or gauge_pass is None) and (renorm_pass or renorm_pass is None)
        }
        
        return results


# Example usage
if __name__ == '__main__':
    # Example: Harmonic oscillator
    print("Example: Testing Harmonic Oscillator Lagrangian")
    print("=" * 60)
    
    # Define Lagrangian: L = (1/2)m*v² - (1/2)k*x²
    m, k = 1.0, 1.0
    omega = np.sqrt(k/m)
    
    def L_harmonic(q, q_dot, t):
        """Harmonic oscillator Lagrangian."""
        return 0.5 * m * q_dot[0]**2 - 0.5 * k * q[0]**2
    
    # Generate a solution path: x(t) = A*cos(ωt)
    t_span = (0, 2*np.pi)
    t = np.linspace(t_span[0], t_span[1], 100)
    A = 1.0
    path = A * np.cos(omega * t).reshape(-1, 1)
    
    # Evaluate selection principles
    evaluator = SelectionPrinciples()
    results = evaluator.evaluate_action_principle(L_harmonic, path, t_span)
    
    print(f"\nAction along true solution: {results['action']:.6f}")
    print(f"Mean action for perturbed paths: {results['mean_perturbation_action']:.6f}")
    print(f"Percentile rank: {results['percentile']:.2f}%")
    print(f"Is minimum: {results['is_minimum']}")
    
    if results['is_minimum']:
        print("\n✅ PASS: Solution satisfies action minimization principle")
    else:
        print("\n❌ FAIL: Solution does not minimize action")
