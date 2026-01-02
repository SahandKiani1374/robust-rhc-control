# FILE: simulation_demo.py
import numpy as np
import matplotlib.pyplot as plt

# IMPORT the function from your other file
from robust_rhc import run_backward_recursion
# %%


# 1. System Parameters (From Example 1)
F = np.array([[1.1, 0, 0], [0, 0, 1.2], [-1.0, 1.0, 0]]) 
G = np.array([[0, 1.0], [1.0, 1.0], [-1.0, 0]])         
H = np.array([[0.7], [0.5], [-0.7]])                    
Ef = np.array([[0.4, 0.5, -0.6]])                       
Eg = np.array([[0.4, -0.4]])                            

# 2. Simulation Settings
Q = np.eye(3) 
R = np.eye(2)
Pn_plus_1 = np.eye(3)
x0 = np.array([[1.0], [-1.0], [0.5]])
N = 70
mu = 1e10

# 3. RUN THE ALGORITHM
# We call the function we wrote in the other file
print("Calculating robust control gains...")
K_list, _ = run_backward_recursion(F, G, H, Ef, Eg, Q, R, Pn_plus_1, N, mu)
print("Calculation complete.")

# 4. Forward Simulation
x_sim = np.zeros((N + 1, 3, 1))
u_sim = np.zeros((N, 2, 1))
x_sim[0] = x0

np.random.seed(42)
for i in range(N):
    xi = x_sim[i]
    ui = K_list[i] @ xi 
    u_sim[i] = ui
    
    # Realized system with random uncertainty
    delta_i = np.random.uniform(-1, 1)
    Fi = F + H * delta_i @ Ef
    Gi = G + H * delta_i @ Eg
    x_sim[i+1] = Fi @ xi + Gi @ ui

# 5. Visualization
plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

t = np.arange(N + 1)
ax1.plot(t, x_sim[:, 0, 0], color='black', ls='--', label='$x_1$')
ax1.plot(t, x_sim[:, 1, 0], color='silver', ls=':', lw=2, label='$x_2$')
ax1.plot(t, x_sim[:, 2, 0], color='black', ls='-.', label='$x_3$')
ax1.set_title("(a) States")
ax1.set_xlabel("i")
ax1.set_ylabel("x")
ax1.grid(True, ls=':', alpha=0.6)
ax1.legend()

ax2.plot(np.arange(N), u_sim[:, 0, 0], color='black', ls='--', label='$u_1$')
ax2.plot(np.arange(N), u_sim[:, 1, 0], color='silver', ls=':', lw=2, label='$u_2$')
ax2.set_title("(b) Control inputs")
ax2.set_xlabel("i")
ax2.set_ylabel("u")
ax2.grid(True, ls=':', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()