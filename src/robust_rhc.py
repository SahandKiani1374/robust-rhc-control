# FILE: robust_rhc.py
import numpy as np

def run_backward_recursion(F, G, H, Ef, Eg, Q, R, P_final, N, mu):
    """
    Performs the backward recursion to find control gains K_list.
    Based on the Robust Receding Horizon Control method.
    """
    
    # --- 1. Setup Dimensions and Augmented Matrices ---
    # We calculate these inside the function so it works for ANY system, not just Example 1
    n = F.shape[0]  # State dimension
    m = G.shape[1]  # Input dimension
    l = H.shape[1]  # Uncertainty dimension (columns of H)

    # Condition: lambda > ||mu H'H||
    # (Calculated automatically based on your script's logic)
    lambda_i = 1.1 * mu * np.linalg.norm(H.T @ H) 

    cal_I = np.vstack([np.eye(n), np.zeros((l, n))])
    cal_G = np.vstack([G, Eg])
    cal_F = np.vstack([F, Ef])

    # Sigma matrix
    S11 = (1.0/mu) * np.eye(n) - (1.0/lambda_i) * (H @ H.T)
    S22 = (1.0/lambda_i) * np.eye(l)
    Sigma = np.block([[S11, np.zeros((n, l))], [np.zeros((l, n)), S22]])

    # --- 2. Backward Recursion Loop ---
    P_list = [None] * (N + 2)
    K_list = [None] * (N + 1)
    P_list[N + 1] = P_final

    for i in range(N, -1, -1):
        Pi_next = P_list[i + 1]
        
        # Building the Augmented Matrix M 
        # (Broken down for readability)
        inv_Pi = np.linalg.inv(Pi_next)
        inv_R = np.linalg.inv(R)
        inv_Q = np.linalg.inv(Q)
        
        # Row blocks construction
        R1 = np.block([inv_Pi, np.zeros((n, m)), np.zeros((n, n)), np.zeros((n, n+l)), np.eye(n), np.zeros((n, m))])
        R2 = np.block([np.zeros((m, n)), inv_R, np.zeros((m, n)), np.zeros((m, n+l)), np.zeros((m, n)), np.eye(m)])
        R3 = np.block([np.zeros((n, n)), np.zeros((n, m)), inv_Q, np.zeros((n, n+l)), np.zeros((n, n)), np.zeros((n, m))])
        R4 = np.block([np.zeros((n+l, n)), np.zeros((n+l, m)), np.zeros((n+l, n)), Sigma, cal_I, -cal_G])
        R5 = np.block([np.eye(n), np.zeros((n, m)), np.zeros((n, n)), cal_I.T, np.zeros((n, n)), np.zeros((n, m))])
        R6 = np.block([np.zeros((m, n)), np.eye(m), np.zeros((m, n)), -cal_G.T, np.zeros((m, n)), np.zeros((m, m))])
        
        M = np.vstack([R1, R2, R3, R4, R5, R6])
        
        # RHS Vector
        b = np.vstack([np.zeros((n, n)), np.zeros((m, n)), -np.eye(n), cal_F, np.zeros((n, n)), np.zeros((m, n))])
        
        # Solution matrix Z
        Z = np.linalg.solve(M, b)
        total_rows = M.shape[0]
        start_K = total_rows - m
        K_list[i] = Z[start_K : total_rows, :] 

        K_list[i] = Z[15:17, :] 
        P_list[i] = Q + cal_F.T @ Z[8:12, :] 

    return K_list, P_list