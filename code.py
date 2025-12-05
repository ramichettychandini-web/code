import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# QUANTUM GATE DEFINITIONS
def hadamard_gate():
    """Hadamard gate matrix"""
    return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def pauli_x():
    """Pauli X (NOT) gate"""
    return np.array([[0, 1], [1, 0]])

def pauli_z():
    """Pauli Z gate"""
    return np.array([[1, 0], [0, -1]])

def controlled_not():
    """CNOT gate"""
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])

def identity(n):
    """Identity matrix of size 2^n x 2^n"""
    return np.eye(2**n)

def tensor_product(matrices):
    """Compute tensor product of multiple matrices"""
    result = matrices[0]
    for mat in matrices[1:]:
        result = np.kron(result, mat)
    return result

def apply_single_qubit_gate(state, gate, qubit, n_qubits):
    """Apply single qubit gate to quantum state"""
    gates = [np.eye(2) for _ in range(n_qubits)]
    gates[qubit] = gate
    operator = tensor_product(gates)
    return operator @ state

def create_superposition(n_qubits):
    """Create equal superposition state"""
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1

    # Apply Hadamard to all qubits
    H = hadamard_gate()
    for i in range(n_qubits):
        state = apply_single_qubit_gate(state, H, i, n_qubits)

    return state

# GROVER'S ALGORITHM IMPLEMENTATION
def create_oracle(n_qubits, marked_state):
    """Create oracle matrix that marks the target state"""
    oracle = np.eye(2**n_qubits, dtype=complex)
    oracle[marked_state, marked_state] = -1
    return oracle

def create_diffusion_operator(n_qubits):
    """Create Grover diffusion operator"""
    size = 2**n_qubits
    diffusion = np.full((size, size), 2/size, dtype=complex)
    diffusion -= np.eye(size)
    return diffusion

def run_grover(n_qubits, marked_state, shots=1024):
    """
    Execute Grover's algorithm simulation
    """
    start_time = time.time()

    # Calculate optimal iterations
    num_iterations = int(np.pi / 4 * np.sqrt(2**n_qubits))

    # Initialize state
    state = create_superposition(n_qubits)

    # Create operators
    oracle = create_oracle(n_qubits, marked_state)
    diffusion = create_diffusion_operator(n_qubits)
    grover_operator = diffusion @ oracle

    # Apply Grover iterations
    gate_count = n_qubits  # Initial Hadamards
    for _ in range(num_iterations):
        state = grover_operator @ state
        gate_count += 2 * n_qubits + 2  # Oracle + Diffusion gates

    # Simulate measurements
    probabilities = np.abs(state)**2
    counts = np.random.multinomial(shots, probabilities)

    runtime = time.time() - start_time

    # Calculate metrics
    success_prob = probabilities[marked_state]
    circuit_depth = num_iterations * 2 + 1

    # Create counts dictionary
    counts_dict = {i: counts[i] for i in range(len(counts)) if counts[i] > 0}

    return {
        'n_qubits': n_qubits,
        'marked_state': marked_state,
        'num_iterations': num_iterations,
        'circuit_depth': circuit_depth,
        'gate_count': gate_count,
        'success_probability': success_prob,
        'runtime': runtime,
        'shots': shots,
        'counts': counts_dict
    }

def gcd(a, b):
    """Greatest common divisor"""
    while b:
        a, b = b, a % b
    return a

def classical_order_finding(a, N):
    """Classical order finding (simulating quantum part)"""
    r = 1
    current = a % N
    while current != 1 and r < N:
        current = (current * a) % N
        r += 1
    return r if current == 1 else None

def shors_algorithm(N):
    """ Shor's algorithm simulation for factoring N uses classical approximation for quantum subroutine """
    start_time = time.time()

    # Check if N is even
    if N % 2 == 0:
        runtime = time.time() - start_time
        return {
            'N': N,
            'factors': [2, N // 2],
            'n_qubits': int(np.ceil(np.log2(N))) * 2,
            'circuit_depth': 10,
            'gate_count': 50,
            'runtime': runtime,
            'success': True
        }

    # Try random values of a
    max_attempts = 10
    for attempt in range(max_attempts):
        a = np.random.randint(2, N)

        # Check if a and N share a common factor
        g = gcd(a, N)
        if g > 1:
            runtime = time.time() - start_time
            return {
                'N': N,
                'factors': [g, N // g],
                'n_qubits': int(np.ceil(np.log2(N))) * 2,
                'circuit_depth': 15,
                'gate_count': 75,
                'runtime': runtime,
                'success': True
            }

        # Find order r
        r = classical_order_finding(a, N)

        if r and r % 2 == 0:
            x = pow(a, r // 2, N)
            if x != N - 1:
                factor1 = gcd(x - 1, N)
                factor2 = gcd(x + 1, N)

                if factor1 > 1 and factor1 < N:
                    runtime = time.time() - start_time
                    return {
                        'N': N,
                        'factors': [factor1, N // factor1],
                        'n_qubits': int(np.ceil(np.log2(N))) * 2,
                        'circuit_depth': 20,
                        'gate_count': 100,
                        'runtime': runtime,
                        'success': True
                    }
                elif factor2 > 1 and factor2 < N:
                    runtime = time.time() - start_time
                    return {
                        'N': N,
                        'factors': [factor2, N // factor2],
                        'n_qubits': int(np.ceil(np.log2(N))) * 2,
                        'circuit_depth': 20,
                        'gate_count': 100,
                        'runtime': runtime,
                        'success': True
                    }

    runtime = time.time() - start_time
    return {
        'N': N,
        'factors': [],
        'n_qubits': int(np.ceil(np.log2(N))) * 2,
        'circuit_depth': 20,
        'gate_count': 100,
        'runtime': runtime,
        'success': False
    }

def run_all_grover_experiments():
    """Run Grover experiments for n=3 to 8 qubits"""
    results_grover = []

    for n in range(3, 9):
        marked_state = 2**(n-1)
        result = run_grover(n, marked_state, shots=1024)
        results_grover.append(result)

    df = pd.DataFrame(results_grover)
    df.to_csv('grover_results.csv', index=False)

    return results_grover

def run_all_shor_experiments():
    """Run Shor's algorithm for factorization"""
    results_shor = []
    test_numbers = [15, 21, 35]

    for N in test_numbers:
        result = shors_algorithm(N)
        results_shor.append(result)

    df = pd.DataFrame(results_shor)
    df.to_csv('shor_results.csv', index=False)

    return results_shor

def plot_grover_circuit():
    """ Grover circuit representation"""
    fig, ax = plt.subplots(figsize=(12, 6))

    n = 4
    stages = ['|0⟩', 'H', 'Oracle', 'Diffusion', 'Oracle', 'Diffusion', 'Measure']

    for i in range(n):
        y = n - i - 1
        ax.plot([0, len(stages)-1], [y, y], 'k-', linewidth=2)
        ax.text(-0.5, y, f'q{i}', fontsize=12, ha='right', va='center')

        for j, stage in enumerate(stages):
            if stage == 'H':
                ax.add_patch(plt.Rectangle((j-0.15, y-0.15), 0.3, 0.3,
                                          fill=True, color='lightblue', edgecolor='black'))
                ax.text(j, y, 'H', fontsize=10, ha='center', va='center', fontweight='bold')
            elif stage in ['Oracle', 'Diffusion']:
                ax.add_patch(plt.Rectangle((j-0.2, y-0.2), 0.4, 0.4,
                                          fill=True, color='lightcoral', edgecolor='black'))
            elif stage == 'Measure':
                ax.plot([j, j], [y-0.2, y+0.2], 'b-', linewidth=2)
                ax.add_patch(plt.Polygon([(j, y+0.2), (j+0.15, y+0.3), (j-0.15, y+0.3)],
                                        fill=True, color='yellow', edgecolor='black'))

    ax.set_xlim(-1, len(stages))
    ax.set_ylim(-0.5, n-0.5)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_yticks([])
    ax.set_title("Grover's Algorithm Circuit (n=4 qubits)", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    return fig

def plot_shor_circuit():
    """ Shor circuit representation"""
    fig, ax = plt.subplots(figsize=(12, 6))

    n = 6
    stages = ['|0⟩', 'H', 'QFT', 'Modular Exp', 'Inverse QFT', 'Measure']

    for i in range(n):
        y = n - i - 1
        ax.plot([0, len(stages)-1], [y, y], 'k-', linewidth=2)
        ax.text(-0.5, y, f'q{i}', fontsize=12, ha='right', va='center')

        for j, stage in enumerate(stages):
            if stage == 'H' and i < n//2:
                ax.add_patch(plt.Rectangle((j-0.15, y-0.15), 0.3, 0.3,
                                          fill=True, color='lightblue', edgecolor='black'))
                ax.text(j, y, 'H', fontsize=10, ha='center', va='center', fontweight='bold')
            elif stage in ['QFT', 'Inverse QFT'] and i < n//2:
                ax.add_patch(plt.Rectangle((j-0.25, y-0.2), 0.5, 0.4,
                                          fill=True, color='lightgreen', edgecolor='black'))
            elif stage == 'Modular Exp':
                ax.add_patch(plt.Rectangle((j-0.25, y-0.2), 0.5, 0.4,
                                          fill=True, color='orange', edgecolor='black'))
            elif stage == 'Measure' and i < n//2:
                ax.plot([j, j], [y-0.2, y+0.2], 'b-', linewidth=2)

    ax.set_xlim(-1, len(stages))
    ax.set_ylim(-0.5, n-0.5)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, fontsize=11)
    ax.set_yticks([])
    ax.set_title("Shor's Algorithm Circuit (simplified)", fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout()
    return fig

def plot_runtime_comparison(results_grover, results_shor):
    """ Runtime comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))

    grover_runtimes = [r['runtime'] for r in results_grover]
    shor_runtimes = [r['runtime'] for r in results_shor]

    x_grover = [r['n_qubits'] for r in results_grover]
    x_shor = [str(r['N']) for r in results_shor]

    x_pos_grover = np.arange(len(x_grover))
    x_pos_shor = np.arange(len(x_shor)) + len(x_grover) + 1

    ax.bar(x_pos_grover, grover_runtimes, width=0.6,
           label='Grover', alpha=0.8, color='#2ecc71')
    ax.bar(x_pos_shor, shor_runtimes, width=0.6,
           label='Shor', alpha=0.8, color='#e74c3c')

    ax.set_xticks(list(x_pos_grover) + list(x_pos_shor))
    ax.set_xticklabels([f'n={x}' for x in x_grover] + [f'N={x}' for x in x_shor])
    ax.set_xlabel('Problem Size', fontsize=12)
    ax.set_ylabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison: Grover vs Shor', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig

def plot_success_iterations(results_grover):
    """ Success probability vs iterations"""
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = [r['num_iterations'] for r in results_grover]
    success_probs = [r['success_probability'] for r in results_grover]
    n_qubits = [r['n_qubits'] for r in results_grover]

    scatter = ax.scatter(iterations, success_probs, s=200, c=n_qubits,
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1.5)

    for i, n in enumerate(n_qubits):
        ax.annotate(f'n={n}', (iterations[i], success_probs[i]),
                   fontsize=9, ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel('Number of Grover Iterations', fontsize=12)
    ax.set_ylabel('Success Probability', fontsize=12)
    ax.set_title('Grover Success Probability vs Iterations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Qubits', fontsize=11)

    plt.tight_layout()
    return fig

def plot_gate_count_heatmap(results_grover):
    """ Gate count heatmap"""
    fig, ax = plt.subplots(figsize=(10, 6))

    data_matrix = []
    labels = []

    for r in results_grover:
        data_matrix.append([r['gate_count'], r['circuit_depth']])
        labels.append(f"n={r['n_qubits']}")

    data_matrix = np.array(data_matrix).T

    sns.heatmap(data_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=labels, yticklabels=['Gate Count', 'Circuit Depth'],
                cbar_kws={'label': 'Count'}, ax=ax, linewidths=0.5)

    ax.set_title('Grover Algorithm: Gate Count and Circuit Depth',
                fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig

def plot_depth_vs_success(results_grover):
    """ Circuit depth vs success probability"""
    fig, ax = plt.subplots(figsize=(10, 6))

    depths = [r['circuit_depth'] for r in results_grover]
    success = [r['success_probability'] for r in results_grover]
    sizes = [r['gate_count'] for r in results_grover]
    n_qubits = [r['n_qubits'] for r in results_grover]

    scatter = ax.scatter(depths, success, s=np.array(sizes)/2,
                        c=n_qubits, cmap='plasma', alpha=0.6,
                        edgecolors='black', linewidth=1.5)

    for i, n in enumerate(n_qubits):
        ax.annotate(f'n={n}', (depths[i], success[i]),
                   fontsize=9, ha='right', va='top', fontweight='bold')

    ax.set_xlabel('Circuit Depth', fontsize=12)
    ax.set_ylabel('Success Probability', fontsize=12)
    ax.set_title('Circuit Complexity vs Success Rate (bubble size = gate count)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Qubits', fontsize=11)

    plt.tight_layout()
    return fig

def generate_all_visuals(results_grover, results_shor):
    """Generate all visualizations"""
    plot_grover_circuit()
    plt.show()

    plot_shor_circuit()
    plt.show()

    plot_runtime_comparison(results_grover, results_shor)
    plt.show()

    plot_success_iterations(results_grover)
    plt.show()

    plot_gate_count_heatmap(results_grover)
    plt.show()

    plot_depth_vs_success(results_grover)
    plt.show()

def merge_results():
    """Merge and analyze results"""
    df_grover = pd.read_csv('grover_results.csv')
    df_shor = pd.read_csv('shor_results.csv')

    df_grover['algorithm'] = 'Grover'
    df_shor['algorithm'] = 'Shor'

    summary = {
        'grover_best_success': df_grover['success_probability'].max(),
        'grover_avg_success': df_grover['success_probability'].mean(),
        'grover_fastest': df_grover['runtime'].min(),
        'grover_slowest': df_grover['runtime'].max(),
        'shor_success_rate': df_shor['success'].sum() / len(df_shor),
        'shor_avg_runtime': df_shor['runtime'].mean()
    }

    pd.DataFrame([summary]).to_csv('final_summary.csv', index=False)


    print("FINAL SUMMARY")

    print(f"Grover - Best Success: {summary['grover_best_success']:.3f}")
    print(f"Grover - Avg Success: {summary['grover_avg_success']:.3f}")
    print(f"Grover - Runtime Range: {summary['grover_fastest']:.4f}s - {summary['grover_slowest']:.4f}s")
    print(f"Shor - Success Rate: {summary['shor_success_rate']:.1%}")
    print(f"Shor - Avg Runtime: {summary['shor_avg_runtime']:.4f}s")


def main():
    """Main execution function"""
    print("Starting Quantum Algorithm Benchmarking...")

    results_grover = run_all_grover_experiments()

    results_shor = run_all_shor_experiments()

    generate_all_visuals(results_grover, results_shor)

    merge_results()

if __name__ == "__main__":
    main()