# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:04:14 2024

@author: shun

alpha = 0, beta = 0を許す
"""

import numpy as np
import myoptimize

"""
パラメータの設定
"""
R = 1 # 地域数
N_bar = 1.0 * R
phi = 0.047
eta = 0.25 * 0.25
l_bar = 1.0
G_bar = 1.0
alpha = 0.15
beta = 0.0845
tau_bar = 0.013400372386313837

"""
alpha=0.15, gamma=1.0: tau_bar = 0.013400372386313837
alpha=0.15, gamma=0.5: tau_bar = 0.02747204923913058
alpha=0.0, gamma=1.0: tau_bar = 0.01604487576023752
alpha=0.0, gamma=0.5: tau_bar = 0.02985602094519399
"""

gam = 1.0
delta_G_base = 1.0 + phi
f_common = 0.09155410014885551062

H_bar = np.ones(R)

if R > 1:
    A_bar = np.linspace(0.9, 1.1, R) * 1.4497217810323038
else:
    A_bar = np.ones(R) * 1.4497217810323038

"""
変数ベクトルの設定
"""
f = np.ones(R) * f_common
delta_G = np.ones(R) * delta_G_base
N = H_bar / np.sum(H_bar) * N_bar
A = np.ones(R)
L = np.ones(R)
Y = np.ones(R)
w = np.ones(R)
rho = np.ones(R)
h = np.ones(R)
H_G = np.ones(R)
H_M = np.ones(R)
P_G = np.ones(R)
G = np.ones(R)
e_I = np.ones(R)
delta_e_I = np.ones(R)
tau = np.ones(R)
u = np.ones(R)
Pi_G = np.ones(R)
V = np.ones(R)
u_bar = np.ones(1)
W = np.ones(1)

"""
均衡を求めるための方程式
"""
print_convergence = False
def equations(X):
    N[0:R] = X[0:R]
    if np.any(N[0:R] <= 0.0):
        raise ValueError
    w[0:R] = X[R:2*R]
    if np.any(w[0:R] <= 0.0):
        raise ValueError
    rho[0:R] = X[2*R:3*R]
    if np.any(rho[0:R] <= 0.0):
        raise ValueError
    u_bar[0] = X[3*R]
    
    if alpha > 0.0:
        P_G[0:R] = 1.0 / G_bar * np.power(1.0 / (1.0 - alpha), 1.0 - alpha) * np.power(rho[0:R] / alpha, alpha)
    else:
        P_G[0:R] = 1.0 / G_bar
    
    h[0:R] = eta / rho[0:R]
    
    G[0:R] = np.power( delta_G[0:R] * w[0:R] * tau_bar * gam / P_G[0:R], 1.0 / (gam + 1.0) ) * N[0:R]
    
    H_G[0:R] = alpha * P_G[0:R] * G[0:R] / rho[0:R]
    
    tau[0:R] = tau_bar * np.power( N[0:R] / G[0:R], gam )
    if np.any(l_bar - tau[0:R] <= 0.0):
        raise ValueError("l_bar is too small.")
    
    L[0:R] = N[0:R] * (l_bar - tau[0:R])
    
    A[0:R] = A_bar[0:R] * np.power(L[0:R], phi)
    
    if beta > 0.0:
        w_alt = A[0:R] * (1.0 - beta) * np.power( beta / rho[0:R], beta / (1.0 - beta) )
    else:
        w_alt = A[0:R]
    
    Y[0:R] = w_alt[0:R] * L[0:R] / (1.0 - beta)
    
    H_M[0:R] = beta * Y[0:R] / rho[0:R]
    
    u[0:R] = w[0:R] * (l_bar - tau[0:R]) - f[0:R] - eta * np.log(rho[0:R]) - eta
    
    vector = np.ones(3*R+1)
    
    vector[0:R] = u[0:R] - u_bar[0]
    
    vector[R:2*R] = np.log( w_alt[0:R] / w[0:R] )
    
    vector[2*R:3*R] = np.log( (N[0:R] * h[0:R] + H_M[0:R] + H_G[0:R]) / H_bar[0:R] )
    
    vector[3*R] = np.log( np.sum(N[0:R]) / N_bar )
    
    if print_convergence:
        print(X, vector)
    
    return vector

"""
均衡を求める
"""
def find_equilibrium(Print_convergence):
    global print_convergence
    print_convergence = Print_convergence
    X = np.zeros(3*R+1)
    X[0:R] = N[0:R]
    X[R:2*R] = w[0:R]
    X[2*R:3*R] = rho[0:R]
    X[3*R] = u_bar[0]
    typx = np.zeros(3*R+1)
    typx[3*R] = 0.1
    myoptimize.solve1(equations, X, typx=typx)
    e_I[0:R] = gam * w[0:R] * tau[0:R] * (1.0 + phi) - phi * w[0:R] * (l_bar - tau[0:R])
    delta_e_I[0:R] = e_I[0:R] / (gam * w[0:R] * tau[0:R])
    Pi_G[0:R] = N[0:R] * f[0:R] - P_G[0:R] * G[0:R]
    V[0:R] = N[0:R] * u[0:R] + rho[0:R] * H_bar[0:R] + Pi_G[0:R]
    W[0] = np.sum(V[0:R])
    print_convergence = False

"""
地価rhoの都市人口Nに対する弾力性を計算する
"""
def compute_E_rho_N():
    global N_bar
    original_N_bar = N_bar
    N_bar = original_N_bar * (1.0 + 1.0e-4)
    find_equilibrium(False)
    rho_plus = rho.copy()
    N_bar = original_N_bar * (1.0 - 1.0e-4)
    find_equilibrium(False)
    rho_minus = rho.copy()
    N_bar = original_N_bar
    find_equilibrium(False)
    original_rho = rho.copy()
    e_I[0:R] = gam * w[0:R] * tau[0:R] * (1.0 + phi) - phi * w[0:R] * (l_bar - tau[0:R])
    delta_e_I[0:R] = e_I[0:R] / (gam * w[0:R] * tau[0:R])
    Pi_G[0:R] = N[0:R] * f[0:R] - P_G[0:R] * G[0:R]
    V[0:R] = N[0:R] * u[0:R] + rho[0:R] * H_bar[0:R] + Pi_G[0:R]
    W[0] = np.sum(V[0:R])
    E_rho_N = (rho_plus - rho_minus) / (2.0 * original_N_bar * 1.0e-4) * original_N_bar / original_rho
    return E_rho_N

"""
Second Bestのdelta_Gを求める
"""
print_convergence2 = False
def objective_function(X):
    delta_G[0:R] = X[0:R]
    if np.any(delta_G[0:R] <= 0.0):
        raise ValueError
    find_equilibrium(False)
    if print_convergence2:
        print(X, W)
    return - W[0]

def find_optimal_delta_G(Print_convergence2):
    global print_convergence2
    print_convergence2 = Print_convergence2
    X = np.zeros(R)
    X[0:R] = delta_G[0:R]
    res = myoptimize.optimize(objective_function, X)
    objective_function(res.x)
    print_convergence2 = False

# ==========================================
# 実行例
# ==========================================
if __name__ == "__main__":
    # 閉鎖都市(R = 1)を想定し、政策パラメータを固定したうえで外生的にN_barを変化させ、
    # 通勤時間の平均値tau_meanと居住の限界外部費用externalityの変化を調べる
    # 政策パラメータは対称なケースの最適値
    """
    if R != 1:
        raise ValueError("R = 1 is assumed.")
    
    delta_G[0] = 1.0 + phi
    with open("result_closed_city.csv", "w") as fout:
        fout.write("N,tau,externality\n")
        
        N_bar_s = np.linspace(0.1, 2.5, 25)
        for i in range(25):
            N_bar = N_bar_s[i]
            print(f"N_bar = {N_bar}")
            find_equilibrium(False)
            fout.write(f"{N_bar_s[i]},{tau[0]},{e_I[0]}\n")
    """
    
    # 閉鎖都市(R = 1)を想定し、
    # 都市0の可処分所得（料金はe_Iとする）、通勤時間、rhoのNに対する弾力性を出力
    """
    find_equilibrium(False)
    print(f"Disposable Income: {w[0] * (l_bar - tau[0]) - e_I[0]}")
    print(f"Commuting Travel Time: {tau[0]}")
    print(f"Elasticity of rho to N: {compute_E_rho_N()[0]}")
    """
    
    # Second Best均衡を求める
    
    find_optimal_delta_G(False) # 最適化過程を表示したい場合はTrueにする
    
    with open("result.csv", "w") as fout:
        fout.write("r,N,delta_G,tau\n")
        for r in range(R):
            fout.write(f"{r},{N[r]},{delta_G[r]},{tau[r]}\n")
    