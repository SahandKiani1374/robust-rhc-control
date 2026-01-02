# Robust Receding Horizon Control for Uncertain Systems

This code implements the robust control framework presented in:

Optimal Robust Linear Quadratic Regulator for Systems Subject to Uncertainties Authors: Marco H. Terra, João P. Cerri, and João Y. Ishihara  IEEE Transactions on Automatic Control, Vol. 59, No. 9, September 2014 


If you use this code for your research, please cite the paper above.

A Python implementation of a **Robust Receding Horizon Control (RHC)** algorithm for discrete-time linear systems subject to norm-bounded structured uncertainties. 

This repository implements the backward recursion and forward simulation methods described in the reference paper listed below, demonstrating a solution to the robust Min-Max control problem.

## 📖 Overview

In real-world control applications, systems often deviate from their nominal models due to parameter drift, modeling errors, or external disturbances. This project implements a control strategy that minimizes a quadratic cost function while accounting for the "worst-case" uncertainty realization.

The algorithm addresses systems of the form:

$$x_{k+1} = (F + H\Delta_k E_f)x_k + (G + H\Delta_k E_g)u_k$$

Where:
- $F, G$ represent the nominal system dynamics.
- $H, E_f, E_g$ define the structure of the uncertainty.
- $\Delta_k$ is the time-varying, norm-bounded uncertainty ($||\Delta_k|| \leq 1$).

## ✨ Features

* **Robust Backward Recursion:** Computes optimal control gains $K_i$ and cost matrices $P_i$ by solving a regularized robust least-squares problem at each time step.
* **Augmented Matrix Solver:** Implements the specific block-matrix inversion technique (Table I of the reference paper) to handle weight inversions and uncertainty penalties efficiently.
* **Stochastic Simulation:** Validates the controller by simulating the closed-loop system under random uncertainty realizations.
* **Visualization:** Generates time-domain plots for system states $x$ and control inputs $u$.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed along with the following dependencies:

* `numpy` (for matrix operations)
* `matplotlib` (for visualization)

### Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/YourUsername/robust-rhc-control.git](https://github.com/YourUsername/robust-rhc-control.git)
    cd robust-rhc-control
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

Run the main simulation script:

```bash
python simulation_demo.py