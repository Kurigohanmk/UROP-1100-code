import numpy as np
import qutip as qt
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt

# Physical parameters in the paper (MHz)
w1 = -2.97 * 2 * np.pi
w2 = -6.46 * 2 * np.pi
Omega1 = 7.91 * 2 * np.pi
Omega2 = -1.39 * 2 * np.pi
gz = 5.92 * 2 * np.pi
gx = 1.39 * 2 * np.pi

# Pauli matrices for the two-qubit system
I2 = qt.qeye(2)
sz1 = qt.tensor(qt.sigmaz(), I2)
sz2 = qt.tensor(I2, qt.sigmaz())
sx1 = qt.tensor(qt.sigmax(), I2)
sx2 = qt.tensor(I2, qt.sigmax())

# Hamiltonians
H0 = 0.5 * (w1 * sz1 + w2 * sz2 + Omega2 * sx2 + gz * sz1 * sz2 + gx * sz1 * sx2)
Hc = Omega1 * sx1
M_obs = sz1 # Observable for measurement

# Generate random control fields (Form 6 in the paper)
def generate_random_field(K=10):
    F = np.random.uniform(0, 1, K)
    F = F / np.sum(F)
    mu = np.random.uniform(0, 4, K) * 2 * np.pi
    phi = np.random.uniform(0, 2 * np.pi, K)
    def f(t, args):
        return np.sum(F * np.cos(mu * t + phi))
    return f

# Create a nontrivial random state using a 0.8 us preparation pulse

tlist_prep = np.linspace(0, 0.8, 100)
prep_field = generate_random_field(K=10)
H_prep = [H0, [Hc, prep_field]]
U_prep = qt.propagator(H_prep, tlist_prep)[-1]

# Initial polarized state |00> (rho_0)
psi_0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
rho_0 = qt.ket2dm(psi_0)

# The target nontrivial state to be reconstructed
rho_target = U_prep * rho_0 * U_prep.dag()
print(f"Target State Concurrence: {qt.concurrence(rho_target):.4f}")

# Tomography Phase
tlist_tomo = np.linspace(0, 0.7, 100)

basis_matrices = []
paulis = [I2, qt.sigmax(), qt.sigmay(), qt.sigmaz()]
for p1 in paulis:
    for p2 in paulis:
        if p1 == I2 and p2 == I2:
            continue
        basis_matrices.append(qt.tensor(p1, p2) / 2.0)

M_matrix = np.zeros((15, 15))
y_record = np.zeros(15)

# 15 random pulses for tomography
for n in range(15):
    f_t = generate_random_field(K=10)
    H_t = [H0, [Hc, f_t]]
    
    U_t = qt.propagator(H_t, tlist_tomo)[-1]
    evolved_M = U_t.dag() * M_obs * U_t
    
    # Simulate measurement on the nontrivial target state
    y_record[n] = np.real((evolved_M * rho_target).tr())
    
    for m in range(15):
        M_matrix[n, m] = np.real((evolved_M * basis_matrices[m]).tr())

# Reconstruct using least squares
res = lsq_linear(M_matrix, y_record)
r_reconstructed = res.x

rho_reconstructed = 0.25 * qt.tensor(I2, I2)
for m in range(15):
    rho_reconstructed += r_reconstructed[m] * basis_matrices[m]

fidelity = qt.fidelity(rho_target, rho_reconstructed)
print(f"Reconstruction Fidelity: {fidelity:.4f}")

# Plot 3D bar charts of absolute values of density matrices
def plot_density_matrix_3d(ax, rho_matrix, title):
    abs_rho = np.abs(rho_matrix.full())
    xpos, ypos = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = dy = 0.6
    dz = abs_rho.flatten()

    # Create 3D bars
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Formatting
    ticks = np.arange(4) + 0.3
    labels = ['|00>', '|01>', '|10>', '|11>']
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_zlim(0, 1)
    ax.set_title(title, fontsize=12, pad=10)
    ax.view_init(elev=35, azim=-45)

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
plot_density_matrix_3d(ax1, rho_target, "Target Random State (Numerical Ideal)")

ax2 = fig.add_subplot(122, projection='3d')
plot_density_matrix_3d(ax2, rho_reconstructed, f"Reconstructed State\nFidelity: {fidelity:.4f}\nTarget State Concurrence: {qt.concurrence(rho_target):.4f}")

plt.show()