# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 08:52:47 2025

@author: segis
"""

import numpy as np
import myoptimize

# ==========================================
# 共通モデルパラメータ設定
# ==========================================
S = 11        # 都市内ゾーン数
H_bar = 1.0 * np.ones(S)   # 各ゾーンの賦存土地面積 (S,)
l_bar = 1.0   # 労働賦存量
eta = 0.25 * 0.25 
theta = 0.6901866790315698

# 生産関数パラメータ（ベースプロファイル）
A_base_profile = np.ones(S)
center = (S-1) / 2
dr = - np.log(0.8) / center
i_s = np.arange(S)
# ベースとなる都市内生産性分布 (S,)
A_base_profile[:] = np.exp( - dr * np.fabs( center - i_s ) )
mean_A_base_profile = np.mean(A_base_profile)
A_base_profile[:] *= 1.5315660867684704 / mean_A_base_profile

phi = 0.047    # 集積の経済
beta = 0.0845  # 生産の土地投入シェア

# 交通・インフラパラメータ
gamma = 1.0   # 移動時間弾力性
alpha = 0.15  # インフラ生産の土地シェア
tau_bar = 0.0025835454076373666  # 基準旅行時間

"""
alpha=0.15, gamma=1.0: tau_bar = 0.0025835454076373666
alpha=0.15, gamma=0.5: tau_bar = 0.006385226537706536
alpha=0.0, gamma=1.0: tau_bar = 0.003151781807372809
alpha=0.0, gamma=0.5: tau_bar = 0.006995495091987303
"""

G_bar = 1.0   # インフラ生産効率

# 数値計算用パラメータ
tol = 1.0e-11  # 収束判定
tol2 = 100 # 最低繰り返し回数
damping = 0.1 # 収束安定化のためのダンピング係数

# ==========================================
# 複数都市モデル設定
# ==========================================
R = 11  # 都市数
N_bar = 1 * R * S # 全都市の総人口

# 都市別の平均基礎的生産性
# (R,)
if R > 1:
    A_scalar_r = np.linspace(0.5, 1.5, R)
else:
    A_scalar_r = np.ones(R)

# 都市ごとのAプロファイル (R, S)
# A_base_profile (S,) を R個に拡張してスカラー倍
A_bar = A_scalar_r[:, None] * A_base_profile[None, :]

# 政策パラメータ
delta_G = (1.0 + phi) * np.ones(R)
f = np.array([phi])

# ==========================================
# モデル主変数設定
# ==========================================

# 都市内シェア (R, S, S)
n_share = np.ones((R, S, S)) / (R * S * S)

# インフラ (R, S)
G = np.ones((R, S))

# 地価 (R, S)
rho = np.ones((R, S))

# ==========================================
# グローバル均衡ソルバー (ベクトル化)
# ==========================================
def solve_global_equilibrium():
    # 距離行列 (S, S) - 全都市共通
    idx = np.arange(S)
    dist_mat = np.abs(idx[:, None] - idx) # |i-j|
    
    # 距離行列 (S, S) - 全都市共通
    idx = np.arange(S)
    dist_mat = np.abs(idx[:, None] - idx).astype(float) # |i-j|
    
    # 同一ゾーン内の移動距離は1/2
    np.fill_diagonal(dist_mat, 0.5)
    
    # 料金行列 (S, S) - 全都市共通
    f_mat = f[0] * dist_mat
    
    min_error = 1.0e100
    n_failure = 0
    
    while True:
        # 1. 居住・就業分布の実数 (R, S, S)
        n_mat = N_bar * n_share
        
        # 2. リンク交通量 Q (R, S)
        # Q_plus: リンク i+ の交通総量
        # Q_minus: リンク i- の交通総量
        
        Q_plus = np.zeros((R, S))
        Q_minus = np.zeros((R, S))
        
        for i in range(S):
            # --- Q_plus (リンク i+) ---
            # n_ii/4 + sum(n_ji/2, j<i) + sum(n_ik/2, k>i) + sum(n_jk, j<i, k>i)
            
            # 1. ゾーン内交通 (n_ii) -> 係数 1/4
            # n_mat[:, i, i] は (R,)
            term1 = n_mat[:, i, i] / 4.0
            
            # 2. 左から来て i で終わる (流入) -> 係数 1/2
            # n_mat[:, :i, i] の和
            term2 = np.sum(n_mat[:, :i, i], axis=1) / 2.0
            
            # 3. i から出て右へ行く (流出) -> 係数 1/2
            # n_mat[:, i, i+1:] の和
            term3 = np.sum(n_mat[:, i, i+1:], axis=1) / 2.0
            
            # 4. i を通過して右へ (通過) -> 係数 1.0
            # n_mat[:, :i, i+1:] の和 (R, i, S-(i+1)) -> axis=(1,2)
            term4 = np.sum(n_mat[:, :i, i+1:], axis=(1, 2))
            
            Q_plus[:, i] = term1 + term2 + term3 + term4
            
            # --- Q_minus (リンク i-) ---
            # n_ii/4 + sum(n_ji/2, j>i) + sum(n_ik/2, k<i) + sum(n_jk, j>i, k<i)
            
            # 1. ゾーン内交通 (n_ii) -> 係数 1/4
            term1_m = n_mat[:, i, i] / 4.0
            
            # 2. 右から来て i で終わる (流入) -> 係数 1/2
            term2_m = np.sum(n_mat[:, i+1:, i], axis=1) / 2.0
            
            # 3. i から出て左へ行く (流出) -> 係数 1/2
            term3_m = np.sum(n_mat[:, i, :i], axis=1) / 2.0
            
            # 4. i を通過して左へ (通過) -> 係数 1.0
            term4_m = np.sum(n_mat[:, i+1:, :i], axis=(1, 2))
            
            Q_minus[:, i] = term1_m + term2_m + term3_m + term4_m
            
        # 3. リンク時間 t (R, S)
        # 各ゾーン i にリンク i+, i- が存在し、そのゾーンの Q と G のみに依存する
        
        # t_plus: (R, S)
        # t_{i+} = tau_bar * (Q_{i+} / G_i)^gamma
        t_plus = tau_bar * np.power(Q_plus / G, gamma)
        
        # t_minus: (R, S)
        # t_{i-} = tau_bar * (Q_{i-} / G_i)^gamma
        t_minus = tau_bar * np.power(Q_minus / G, gamma)
        
        # 4. OD間所要時間 tau (R, S, S)
        
        # 累積和 (Cumulative Sum) の計算
        # cum_t_plus[:, k] = sum(t_plus[:, 0:k+1])
        cum_t_plus = np.cumsum(t_plus, axis=1)
        cum_t_minus = np.cumsum(t_minus, axis=1)
        
        tau_mat = np.zeros((R, S, S))
        
        # --- 対角成分 (i=j) ---
        # tau_ii = 0.5 * (t_{i+}/2 + t_{i-}/2) = 0.25 * (t_{i+} + t_{i-})
        diag_val = 0.25 * (t_plus + t_minus) # (R, S)
        for s in range(S):
            tau_mat[:, s, s] = diag_val[:, s]
            
        # --- 上側三角行列 (i < j) : Link + を利用 ---
        # diff_plus[r, i, j] = cum_t_plus[r, j] - cum_t_plus[r, i]
        # これにより sum_{k=i+1}^{j} t_{k+} が得られる
        diff_plus = cum_t_plus[:, None, :] - cum_t_plus[:, :, None]
        
        # 第2ケース:
        # (sum_{k=i+1}^{j} t_{k+}) - 0.5*t_{j+} + 0.5*t_{i+}
        # = (diff_plus) - 0.5*t_{j+} + 0.5*t_{i+}
        term_j_p = 0.5 * t_plus[:, None, :]
        term_i_p = 0.5 * t_plus[:, :, None]
        tau_upper = diff_plus - term_j_p + term_i_p
        
        # --- 下側三角行列 (i > j) : Link - を利用 ---
        # diff_minus[r, i, j] = cum_t_minus[r, i] - cum_t_minus[r, j]
        # これにより sum_{k=j+1}^{i} t_{k-} が得られる
        diff_minus = cum_t_minus[:, :, None] - cum_t_minus[:, None, :]
        
        # 第3ケース:
        # (sum_{k=j+1}^{i} t_{k-}) - 0.5*t_{i-} + 0.5*t_{j-}
        # = (diff_minus) - 0.5*t_{i-} + 0.5*t_{j-}
        term_i_m = 0.5 * t_minus[:, :, None]
        term_j_m = 0.5 * t_minus[:, None, :]
        tau_lower = diff_minus - term_i_m + term_j_m
        
        # マスクを使って代入
        mask_u = np.triu(np.ones((S, S), dtype=bool), k=1)
        mask_l = np.tril(np.ones((S, S), dtype=bool), k=-1)
        
        # ブロードキャストで代入
        tau_mat[:, mask_u] = tau_upper[:, mask_u]
        tau_mat[:, mask_l] = tau_lower[:, mask_l]
        
        # 5. 労働供給と生産 (都市内経済)
        labor_time = l_bar - tau_mat
        
        # 居住者数 n_I (R, S)
        n_I = np.sum(n_mat, axis=2)
        
        # 労働供給 L_j (R, S)
        # sum_i (n_{ij} * labor_{ij})
        L = np.sum(n_mat * labor_time, axis=1)
        L = np.fmax(L, 1.0e-15)
        
        # 生産性 A (R, S)
        A = A_bar * np.power(L, phi)
        
        # 賃金 w (R, S)
        # w = A * (1-beta) * (beta/rho)^(beta/(1-beta))
        w = A * (1.0 - beta) * np.power(beta / rho, beta / (1.0 - beta))
        
        # 産出 Y (R, S)
        Y = w * L / (1.0 - beta)
        
        # 企業土地需要 H_M (R, S)
        H_M = beta * Y / rho
        
        # 6. 時間価値重み付き交通量 U (R, S)
        # 基本ロジックは Q と同じだが、集計対象が n_mat ではなく nw_mat になる
        
        # nw_mat (R, S, S) = n_mat * w_j
        # w: (R, S) -> (R, 1, S) にブロードキャストして掛ける
        nw_mat = n_mat * w[:, None, :]
        
        U_plus = np.zeros((R, S))
        U_minus = np.zeros((R, S))
        
        for i in range(S):
            # --- U_plus (リンク i+) ---
            
            # 1. ゾーン内 (nw_ii) -> 係数 1/4
            term1 = nw_mat[:, i, i] / 4.0
            
            # 2. 流入 (左から来て i で終わる) -> 係数 1/2
            term2 = np.sum(nw_mat[:, :i, i], axis=1) / 2.0
            
            # 3. 流出 (i から出て右へ行く) -> 係数 1/2
            term3 = np.sum(nw_mat[:, i, i+1:], axis=1) / 2.0
            
            # 4. 通過 (i を通過して右へ) -> 係数 1.0
            term4 = np.sum(nw_mat[:, :i, i+1:], axis=(1, 2))
            
            U_plus[:, i] = term1 + term2 + term3 + term4
            
            # --- U_minus (リンク i-) ---
            
            # 1. ゾーン内 (nw_ii) -> 係数 1/4
            term1_m = nw_mat[:, i, i] / 4.0
            
            # 2. 流入 (右から来て i で終わる) -> 係数 1/2
            term2_m = np.sum(nw_mat[:, i+1:, i], axis=1) / 2.0
            
            # 3. 流出 (i から出て左へ行く) -> 係数 1/2
            term3_m = np.sum(nw_mat[:, i, :i], axis=1) / 2.0
            
            # 4. 通過 (i を通過して左へ) -> 係数 1.0
            term4_m = np.sum(nw_mat[:, i+1:, :i], axis=(1, 2))
            
            U_minus[:, i] = term1_m + term2_m + term3_m + term4_m
            
        # 7. 効用
        # u_mat (R, S, S)
        # w_j * labor - f - eta * ln(rho_i)
        term_wage = w[:, None, :] * labor_time
        term_rho = eta * np.log(rho)[:, :, None]
        # f_mat (S, S) は (1, S, S) にブロードキャスト
        u_mat = term_wage - f_mat[None, :, :] - term_rho
        
        # 8. Logit Choice
        # オーバーフロー対策
        u_max = np.max(u_mat[:,:,:])
        exp_u = np.exp(theta * (u_mat[:,:,:] - u_max))
        sum_exp_u = np.sum(exp_u[:,:,:])
        n_share_new = exp_u[:,:,:] / sum_exp_u
        log_sum_exp_u = u_max + np.log(sum_exp_u) / theta
        
        # 9. インフラ供給更新
        if alpha > 0.0:
            P_G = (1.0/G_bar) * ( 1.0 / (1.0 - alpha) ) ** (1.0 - alpha) * np.power(rho / alpha, alpha)
        else:
            P_G = np.ones((R, S)) * (1.0/G_bar)
            
        H_G = alpha * P_G * G / rho
        
        # G_new の計算
        # G_i = [ (tau_bar * gamma * delta_G) / P_G_i * (U_i+ * Q_i+^gamma + U_i- * Q_i-^gamma) ] ^ (1/(1+gamma))
        
        # 右辺のカッコ内の項: (U_{i+} Q_{i+}^\gamma + U_{i-} Q_{i-}^\gamma)
        term_traffic = U_plus * np.power(Q_plus, gamma) + U_minus * np.power(Q_minus, gamma)
        
        # 係数部分: (tau_bar * gamma * delta_G) / P_G
        # delta_G (R,) を (R, S) にブロードキャストするために [:, None] を使用
        coeff = (tau_bar * gamma * delta_G[:, None]) / P_G
        
        # 全体
        base_term = coeff * term_traffic
        
        # べき乗 1/(1+gamma)
        G_new = np.power(base_term, 1.0 / (1.0 + gamma))
        
        # 10. 土地需要
        total_land_demand = n_I[:,:] * eta / rho[:,:] + H_M[:,:] + H_G[:,:]
        rho_new = rho[:,:] * total_land_demand[:,:] / H_bar[None, :]
        
        # --- 収束判定 & ダンピング更新 ---
        error = 0.0
        
        error_tmp = np.max(n_share_new[:,:,:] - n_share[:,:,:])
        error = max(error, error_tmp)
        n_share[:,:] = damping * n_share_new[:,:,:] + (1.0 - damping) * n_share[:,:,:]
        
        error_tmp = np.max(rho_new[:,:] / rho[:,:] - 1.0)
        error = max(error, error_tmp)
        rho[:] = damping * rho_new[:] + (1.0 - damping) * rho[:]
        
        error_tmp = np.max(G_new[:] / G[:] - 1.0)
        error = max(error, error_tmp)
        G[:] = damping * G_new[:] + (1.0 - damping) * G[:]
        
        if error <= tol and error * (1.0 + tol) >= min_error:
            n_failure += 1
            if n_failure == tol2:
                break
        else:
            n_failure = 0
        min_error = min(error, min_error)
    
    # 均衡結果から社会的総余剰Wを計算する
    
    # 都市別地代収入
    # rho * H_bar
    LandRent = np.sum(rho[:,:] * H_bar[None,:], axis=(1))
    
    # 都市別料金収入
    Revenue = np.sum(n_mat[:,:,:] * f_mat[None, :, :], axis=(1,2))
    # 都市別インフラ費用
    Cost = np.sum(P_G[:,:] * G[:,:], axis=1)
    # 都市別政府余剰 (Pi_G)
    Pi_G = Revenue - Cost
    
    # 都市内人口
    N = np.sum(n_I[:,:], axis=1)
    
    W = N_bar * log_sum_exp_u + np.sum(LandRent) + np.sum(Pi_G)
    
    return {
        'rho': rho.copy(), 'w': w, 'n_I': n_I, 'L': L, 'N': N,
        'Q_plus': Q_plus, 'Q_minus': Q_minus,
        't_plus': t_plus, 't_minus': t_minus,
        'P_G': P_G, 'H_G': H_G, 'G': G.copy(),
        'tau_mat': tau_mat, 'f_mat': f_mat,
        'log_sum_exp_u': log_sum_exp_u, 'Revenue': Revenue, 'W': W
    }

# ==========================================
# 最適化ラッパー
# ==========================================
def find_optimal_policy(print_convergence):
    # 指定されたNに対して、Welfareを最大化する (f, delta_G) を求める
    print("Searching optimal policy...")
    
    X = np.ones(R+1)
    X[0:R] = delta_G[0:R]
    X[R] = f[0]
    
    def objective_function(X_local):
        delta_G[0:R] = X_local[0:R]
        f[0] = X_local[R]
        
        # 制約: delta_Gは正
        if np.any(delta_G <= 0.0):
            raise ValueError
            
        res_inner = solve_global_equilibrium()
        W = res_inner["W"]
        if print_convergence:
            print(X_local, W)
        return -W # 最大化のため符号反転
    
    # 最適化実行
    typX = np.zeros(R+1)
    typX[R] = 0.1
    result = myoptimize.optimize(objective_function, X, typx=typX)
    
    delta_G[0:R] = result.x[0:R]
    f[0] = result.x[R]
    
    print("Optimization finished.")
    print(f"Optimal delta_G: {delta_G}")
    print(f"Optimal f: {f[0]}")
    
    # 最終的な収束計算
    res_final = solve_global_equilibrium()
    return res_final

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
    
    delta_G[0] = 1.047
    f[0] = 0.0
    
    with open("result_city_structure_closed_city.csv", "w") as fout:
        fout.write("N/S,tau,externality\n")
        
        N_bar_s = np.linspace(0.1, 2.5, 25) * S
        for i in range(25):
            N_bar = N_bar_s[i]
            print(f"N_bar = {N_bar}")
            res = solve_global_equilibrium()
            tau_mean = np.sum(n_share * res["tau_mat"], axis=(1, 2)) / np.sum(n_share, axis=(1, 2))
            log_sum_exp_u = res['log_sum_exp_u']
            
            N_bar = N_bar_s[i] * (1.0 + 1.0e-4)
            res_plus = solve_global_equilibrium()
            W_plus = res_plus["W"]
            
            N_bar = N_bar_s[i] * (1.0 - 1.0e-4)
            res_minus = solve_global_equilibrium()
            W_minus = res_minus["W"]
            
            externality = log_sum_exp_u - (W_plus - W_minus) / (2.0e-4 * N_bar_s[i])
            
            fout.write(f"{N_bar_s[i]/S},{tau_mean[0]},{externality}\n")
    """
    
    # Second Best均衡を求める
    res = find_optimal_policy(False) # 最適化過程を表示したい場合はTrueにする
    
    w_tau_mean = np.sum( n_share * (res["w"][:, None, :] * res["tau_mat"]) ) / np.sum(n_share)
    print(f"Mean Commuting Travel Time Elasticity: {theta * w_tau_mean}")
    disposable_income_mean = np.sum( n_share * (res["w"][:, None, :] * (l_bar - res["tau_mat"]) - res["f_mat"][None, :, :] ) ) / np.sum(n_share)
    print(f"Mean Disposable Income: {disposable_income_mean}")
    tau_mean = np.sum(n_share * res["tau_mat"], axis=(1, 2)) / np.sum(n_share, axis=(1, 2))
    print(f"Mean Commuting Travel Time: {tau_mean[0]}")
    N = res['N']
    
    with open("result_city_structure.csv", "w") as fout:
        fout.write(f"f,{f[0]}\n\n")
        fout.write("r,N/S,delta_G,tau\n")
        for r in range(R):
            fout.write(f"{r},{N[r]/S},{delta_G[r]},{tau_mean[r]}\n")
    