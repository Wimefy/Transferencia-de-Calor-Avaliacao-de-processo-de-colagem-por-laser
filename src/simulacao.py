# -*- coding: utf-8 -*-
"""
PME 3361 - Transferencia de Calor (Projeto I)
Avaliacao do processo de colagem de pelicula plastica por laser em chapa de aco.
Grupo 15C: h = 150 W/(m^2.K), q'' = 80000 W/m^2, T_inf = 25 C.

Metodo numerico: Diferencas finitas centradas de 2a ordem no espaco com
condicao de simetria de Lomax & Pulliam (p. 44) e integracao temporal
por Euler Implicito com fatoracao LU esparsa (splu).
"""

import json
import os
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import splu

# =============================================================================
# Parametros fisicos e geometricos fixos do problema
# =============================================================================
RHO = 7850.0       # Massa especifica do aco [kg/m^3]
CP = 435.0         # Calor especifico do aco [J/(kg.K)]
K_ACO = 60.0       # Condutividade termica do aco [W/(m.K)]
D_CHAPA = 1.25e-3  # Espessura da chapa [m]
W1 = 44.0e-3       # Largura da pelicula plastica [m]
W2 = 132.0e-3      # Largura total da chapa [m]
L_MEIO = W2 / 2.0  # Semilargura da chapa [m] (0,066 m)
RAZAO_P = W2 / W1  # Razao p = w2/w1 = 3,0
ALPHA = K_ACO / (RHO * CP)  # Difusividade termica [m^2/s]

DT_LASER_PADRAO = 10.0  # Duracao do pulso do laser [s]
T_FINAL_PADRAO = 30.0   # Tempo total de simulacao [s]


def formatar_virgula(valor: float, _pos=None) -> str:
    """Formata numeros em eixos graficos usando virgula decimal."""
    if abs(valor - round(valor)) < 1e-9:
        return f"{int(round(valor))}"
    return f"{valor:.1f}".replace(".", ",")


def resolver_placa(
    h: float,
    q_flux: float | Callable[[float], float],
    T_inf: float,
    N: int = 200,
    Nt: int = 3000,
    t_final: float = T_FINAL_PADRAO,
    dt_laser: float = DT_LASER_PADRAO,
    w_laser: float = W1,
    T_init: float | None = None,
) -> dict:
    """
    Resolve a equacao adimensional de conducao-conveccao 1D transiente na meia
    chapa (0 <= x <= L_MEIO) pelo metodo de Euler Implicito.

    Parametros:
        h        : coeficiente de conveccao [W/(m^2.K)]
        q_flux   : fluxo de calor constante [W/m^2] ou funcao q(t)
        T_inf    : temperatura ambiente [C]
        N        : numero de nos internos da malha espacial
        Nt       : numero de passos de tempo
        t_final  : tempo final de simulacao [s]
        dt_laser : duracao do pulso de aquecimento [s]
        w_laser  : largura total do feixe de laser [m]
        T_init   : temperatura inicial uniforme da chapa [C] (padrao: T_inf)

    Retorna:
        Dicionario com malha espacial, vetor temporal, matriz T(x,t) e
        parametros adimensionais do caso.
    """
    # Parametros adimensionais conforme notas de aula (Prof. Ernani V. Volpe)
    Bi_y = h * D_CHAPA / K_ACO
    Bi = h * (L_MEIO ** 2) / (K_ACO * D_CHAPA)

    t_tilde_final = ALPHA * t_final / (L_MEIO ** 2)
    t_tilde_L = ALPHA * dt_laser / (L_MEIO ** 2)
    dt_tilde = t_tilde_final / Nt
    dt_dim = t_final / Nt

    # Malha espacial adimensional: nos internos j = 1..N em 0 < x_tilde < 1
    dx_tilde = 1.0 / (N + 1)
    x_int = np.arange(1, N + 1, dtype=float) * dx_tilde

    # Matriz tridiagonal A (N x N) da derivada segunda espacial
    inv_dx2 = 1.0 / (dx_tilde ** 2)
    diag_main = np.full(N, -2.0 * inv_dx2)
    diag_sup = np.full(N - 1, 1.0 * inv_dx2)
    diag_inf = np.full(N - 1, 1.0 * inv_dx2)

    # Condicao de simetria em j = 1 (Lomax & Pulliam, p. 44)
    diag_main[0] = (-2.0 / 3.0) * inv_dx2
    diag_sup[0] = (2.0 / 3.0) * inv_dx2

    A = diags(
        [diag_inf, diag_main, diag_sup],
        offsets=[-1, 0, 1],
        format="csc",
    )

    # Matriz do sistema implicito: M = [(1 + 2*Bi*dt_tilde)*I - dt_tilde*A]
    I_mat = eye(N, format="csc")
    M = (1.0 + 2.0 * Bi * dt_tilde) * I_mat - dt_tilde * A
    M = csc_matrix(M)

    # Fatoracao LU esparsa realizada uma unica vez fora do laco temporal
    lu = splu(M)

    # Mascara espacial da regiao irradiada pelo feixe de laser (|x_tilde| < w_laser/w2)
    razao_laser = w_laser / W2
    mask_laser = (x_int < razao_laser - 1e-12).astype(float)

    # Condicao inicial adimensional
    if T_init is None:
        theta = np.zeros(N, dtype=float)
    else:
        theta = np.full(N, (T_init - T_inf) / T_inf, dtype=float)

    # Vetores completos incluindo contorno x_tilde = 0 e x_tilde = 1
    x_full_tilde = np.concatenate(([0.0], x_int, [1.0]))
    x_full = x_full_tilde * L_MEIO
    t_vec = np.linspace(0.0, t_final, Nt + 1)
    T_hist = np.zeros((N + 2, Nt + 1), dtype=float)

    # Reconstrucao de t = 0
    theta_0 = (4.0 * theta[0] - theta[1]) / 3.0
    theta_full = np.concatenate(([theta_0], theta, [0.0]))
    T_hist[:, 0] = T_inf * (1.0 + theta_full)

    # Fator de escala para q_tilde
    fator_q = (L_MEIO ** 2) / (K_ACO * D_CHAPA * T_inf)
    q_const_val = q_flux(0.0) if callable(q_flux) else float(q_flux)

    # Laco temporal
    for step in range(1, Nt + 1):
        t_curr = step * dt_dim
        t_tilde_curr = step * dt_tilde

        if callable(q_flux):
            q_curr = float(q_flux(t_curr))
        else:
            q_curr = float(q_flux) if t_tilde_curr <= t_tilde_L + 1e-12 else 0.0

        q_tilde_val = q_curr * fator_q
        rhs = theta + dt_tilde * (q_tilde_val * mask_laser)
        theta = lu.solve(rhs)

        # Reconstrucao de 2a ordem no centro (x = 0) e Dirichlet em x = L
        theta_0 = (4.0 * theta[0] - theta[1]) / 3.0
        theta_full = np.concatenate(([theta_0], theta, [0.0]))
        T_hist[:, step] = T_inf * (1.0 + theta_full)

    return {
        "x": x_full,
        "x_tilde": x_full_tilde,
        "t": t_vec,
        "T": T_hist,
        "h": h,
        "q_flux_nom": q_const_val,
        "T_inf": T_inf,
        "N": N,
        "Nt": Nt,
        "Bi_y": Bi_y,
        "Bi": Bi,
        "q_tilde": q_const_val * fator_q,
        "alpha": ALPHA,
        "t_tilde_L": t_tilde_L,
        "t_tilde_final": t_tilde_final,
        "dt_tilde": dt_tilde,
        "dx_tilde": dx_tilde,
        "w_laser": w_laser,
    }


def _tempo_cruzamento(t: np.ndarray, y: np.ndarray, alvo: float) -> float | None:
    """Calcula por interpolacao linear o primeiro instante em que y >= alvo."""
    idx = np.where(y >= alvo)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    y0, y1 = float(y[i - 1]), float(y[i])
    t0, t1 = float(t[i - 1]), float(t[i])
    if abs(y1 - y0) < 1e-15:
        return t1
    frac = (alvo - y0) / (y1 - y0)
    return t0 + frac * (t1 - t0)


def _duracao_acima_limiar(t: np.ndarray, T_min: np.ndarray, T_max: np.ndarray,
                          T_lo: float, T_hi: float) -> float:
    """
    Calcula o tempo total (por integracao trapezoidal/sub-intervalos lineares)
    em que T_min(t) >= T_lo e T_max(t) <= T_hi simultaneamente.
    """
    total = 0.0
    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])
        # Discretizacao fina dentro do passo para capturar cruzamentos com exatidao
        s = np.linspace(0.0, 1.0, 21)
        tmin_sub = T_min[i] + s * (T_min[i + 1] - T_min[i])
        tmax_sub = T_max[i] + s * (T_max[i + 1] - T_max[i])
        ok = (tmin_sub >= T_lo) & (tmax_sub <= T_hi)
        # Media nos subsegmentos
        frac = 0.5 * (ok[:-1].astype(float) + ok[1:].astype(float))
        total += float(np.mean(frac)) * dt
    return total


def calcular_metricas(
    res: dict,
    w1: float = W1,
    dt_laser: float = DT_LASER_PADRAO,
    T_lo: float = 90.0,
    T_hi: float = 200.0,
) -> dict:
    """Extrai metricas criticas de temperatura e janela de colagem na pelicula."""
    x = res["x"]
    t = res["t"]
    T = res["T"]

    x_borda = w1 / 2.0
    mask_x_pel = x <= x_borda + 1e-12

    # Serie temporal na borda x = w1/2 (interpolacao exata se no coincidir)
    T_centro_t = T[0, :]
    T_borda_t = np.array([np.interp(x_borda, x, T[:, j]) for j in range(len(t))])

    # Maximo e minimo na regiao da pelicula |x| <= w1/2 ao longo do tempo
    T_pel_nos = T[mask_x_pel, :]
    T_max_pel_t = np.maximum(np.max(T_pel_nos, axis=0), T_borda_t)
    T_min_pel_t = np.minimum(np.min(T_pel_nos, axis=0), T_borda_t)

    # Restringe ao intervalo do pulso [0, dt_laser]
    mask_pulso = t <= dt_laser + 1e-12
    t_pulso = t[mask_pulso]
    idx_10s = int(np.argmin(np.abs(t - dt_laser)))

    T_max_pulso = float(np.max(T_max_pel_t[mask_pulso]))
    T_max_global = float(np.max(T))
    idx_max = int(np.argmax(T_max_pel_t[mask_pulso]))
    t_Tmax = float(t_pulso[idx_max])

    T_centro_10s = float(T_centro_t[idx_10s])
    T_borda_10s = float(T_borda_t[idx_10s])
    T_11mm_10s = float(np.interp(11.0e-3, x, T[:, idx_10s]))
    T_44mm_10s = float(np.interp(44.0e-3, x, T[:, idx_10s]))

    t_90_centro = _tempo_cruzamento(t_pulso, T_centro_t[mask_pulso], T_lo)
    t_90_borda = _tempo_cruzamento(t_pulso, T_min_pel_t[mask_pulso], T_lo)

    tempo_janela_pulso = _duracao_acima_limiar(
        t_pulso, T_min_pel_t[mask_pulso], T_max_pel_t[mask_pulso], T_lo, T_hi
    )
    tempo_janela_total = _duracao_acima_limiar(
        t, T_min_pel_t, T_max_pel_t, T_lo, T_hi
    )

    violou_200 = bool(T_max_pulso > T_hi)
    atingiu_90_borda = bool(t_90_borda is not None)
    margem_200 = T_hi - T_max_pulso

    return {
        "T_centro_t": T_centro_t,
        "T_borda_t": T_borda_t,
        "T_max_pel_t": T_max_pel_t,
        "T_min_pel_t": T_min_pel_t,
        "T_max_pulso": T_max_pulso,
        "T_max_global": T_max_global,
        "t_Tmax": t_Tmax,
        "T_centro_10s": T_centro_10s,
        "T_borda_10s": T_borda_10s,
        "delta_T_pel_10s": T_centro_10s - T_borda_10s,
        "T_11mm_10s": T_11mm_10s,
        "T_44mm_10s": T_44mm_10s,
        "t_90_centro": t_90_centro,
        "t_90_borda": t_90_borda,
        "atingiu_90_borda": atingiu_90_borda,
        "tempo_janela_pulso": tempo_janela_pulso,
        "tempo_janela_total": tempo_janela_total,
        "violou_200": violou_200,
        "margem_200": margem_200,
    }


def validar_benchmark_11A() -> dict:
    """
    Executa validacao obrigatoria para o Grupo 11A:
    h = 100 W/(m^2.K), q'' = 75000 W/m^2, T_inf = 25 C, N = 200, Nt = 3000.
    """
    res = resolver_placa(h=100.0, q_flux=75000.0, T_inf=25.0, N=200, Nt=3000)
    met = calcular_metricas(res)

    print("=" * 72)
    print("VALIDACAO BENCHMARK - GRUPO 11A (N=200, Nt=3000)")
    print("=" * 72)
    print(f"Bi_y             = {res['Bi_y']:.6e}")
    print(f"Bi               = {res['Bi']:.6f}   (esperado: 5.808000)")
    print(f"q_tilde          = {res['q_tilde']:.6f} (esperado: 174.240000)")
    print(f"T_max na pelicula= {met['T_max_pulso']:.4f} C (esperado: ~152.2 C)")
    print(f"t_90 na borda    = {met['t_90_borda']:.4f} s (esperado: ~9.30 s)")
    print(f"T_borda (t=10 s) = {met['T_borda_10s']:.4f} C (esperado: ~93.8 C)")

    assert abs(res["Bi"] - 5.808) < 1e-6, f"Bi divergente: {res['Bi']}"
    assert abs(res["q_tilde"] - 174.240) < 1e-6, f"q_tilde divergente: {res['q_tilde']}"
    assert abs(met["T_max_pulso"] - 152.2) < 0.3, f"T_max divergente: {met['T_max_pulso']}"
    assert met["t_90_borda"] is not None and abs(met["t_90_borda"] - 9.30) < 0.08, (
        f"t_90_borda divergente: {met['t_90_borda']}"
    )
    assert abs(met["T_borda_10s"] - 93.8) < 0.3, f"T_borda_10s divergente: {met['T_borda_10s']}"
    print("STATUS BENCHMARK 11A: APROVADO COM SUCESSO!")
    print("=" * 72 + "\n")
    return {"res": res, "met": met}


def estudo_convergencia_malha(h: float, q_flux: float, T_inf: float) -> dict:
    """Executa estudo de independencia de malha espacial (N) e temporal (Nt)."""
    malhas_N = [100, 200, 400]
    resultados_N = []
    for N_val in malhas_N:
        res = resolver_placa(h=h, q_flux=q_flux, T_inf=T_inf, N=N_val, Nt=3000)
        met = calcular_metricas(res)
        resultados_N.append({
            "N": N_val,
            "Nt": 3000,
            "T_centro_10s": met["T_centro_10s"],
            "T_borda_10s": met["T_borda_10s"],
        })

    ref_N = resultados_N[-1]
    for item in resultados_N:
        item["erro_centro_pct"] = (
            abs(item["T_centro_10s"] - ref_N["T_centro_10s"]) / ref_N["T_centro_10s"] * 100.0
        )
        item["erro_borda_pct"] = (
            abs(item["T_borda_10s"] - ref_N["T_borda_10s"]) / ref_N["T_borda_10s"] * 100.0
        )

    passos_Nt = [1500, 3000, 6000]
    resultados_Nt = []
    for Nt_val in passos_Nt:
        res = resolver_placa(h=h, q_flux=q_flux, T_inf=T_inf, N=200, Nt=Nt_val)
        met = calcular_metricas(res)
        resultados_Nt.append({
            "N": 200,
            "Nt": Nt_val,
            "T_centro_10s": met["T_centro_10s"],
            "T_borda_10s": met["T_borda_10s"],
        })

    ref_Nt = resultados_Nt[-1]
    for item in resultados_Nt:
        item["erro_centro_pct"] = (
            abs(item["T_centro_10s"] - ref_Nt["T_centro_10s"]) / ref_Nt["T_centro_10s"] * 100.0
        )
        item["erro_borda_pct"] = (
            abs(item["T_borda_10s"] - ref_Nt["T_borda_10s"]) / ref_Nt["T_borda_10s"] * 100.0
        )

    print("=" * 72)
    print("ESTUDO DE CONVERGENCIA DE MALHA - GRUPO 15C")
    print("=" * 72)
    print("Convergencia Espacial (Nt = 3000, referencia N = 400):")
    for r in resultados_N:
        print(
            f"  N={r['N']:3d} | T(0,10s)={r['T_centro_10s']:.4f} C (err={r['erro_centro_pct']:.4f}%) | "
            f"T(22mm,10s)={r['T_borda_10s']:.4f} C (err={r['erro_borda_pct']:.4f}%)"
        )
    print("Convergencia Temporal (N = 200, referencia Nt = 6000):")
    for r in resultados_Nt:
        print(
            f"  Nt={r['Nt']:4d} | T(0,10s)={r['T_centro_10s']:.4f} C (err={r['erro_centro_pct']:.4f}%) | "
            f"T(22mm,10s)={r['T_borda_10s']:.4f} C (err={r['erro_borda_pct']:.4f}%)"
        )
    print("=" * 72 + "\n")
    return {"espacial": resultados_N, "temporal": resultados_Nt}


def simular_solucoes_engenharia(h: float, q_base: float, T_inf: float) -> dict:
    """
    Simula e avalia solucoes praticas de engenharia para adequacao do processo
    do Grupo 15C (h=150, q''=80000, T_inf=25).
    """
    print("=" * 72)
    print("VARREDURAS EXPLORATORIAS E SOLUCOES DE ENGENHARIA - GRUPO 15C")
    print("=" * 72)

    # Varredura Solucao A: ajuste de fluxo q'' mantendo w_laser = 44 mm
    print("\n[Varredura Solucao A: Ajuste de q'' (w_laser = 44 mm)]")
    for q_cand in [85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000]:
        r = resolver_placa(h=h, q_flux=float(q_cand), T_inf=T_inf, N=200, Nt=3000)
        m = calcular_metricas(r)
        t90b = f"{m['t_90_borda']:.2f} s" if m["t_90_borda"] is not None else "N/A"
        print(
            f"  q''={q_cand:6d} W/m^2 | T_max={m['T_max_pulso']:6.2f} C | "
            f"T_borda(10s)={m['T_borda_10s']:6.2f} C | t_90_borda={t90b:>7s} | "
            f"Janela={m['tempo_janela_pulso']:.2f} s | Violou200={m['violou_200']}"
        )

    # Varredura Solucao B: alargamento do feixe mantendo q'' = 80000 W/m^2
    print("\n[Varredura Solucao B: Alargamento do feixe w_laser (q'' = 80000 W/m^2)]")
    for w_mm in [48, 52, 54, 56, 60, 64, 66]:
        r = resolver_placa(
            h=h, q_flux=q_base, T_inf=T_inf, N=200, Nt=3000, w_laser=w_mm * 1e-3
        )
        m = calcular_metricas(r)
        t90b = f"{m['t_90_borda']:.2f} s" if m["t_90_borda"] is not None else "N/A"
        print(
            f"  w_laser={w_mm:2d} mm | T_max={m['T_max_pulso']:6.2f} C | "
            f"T_borda(10s)={m['T_borda_10s']:6.2f} C | dT_pel={m['delta_T_pel_10s']:5.2f} C | "
            f"t_90_borda={t90b:>7s} | Janela={m['tempo_janela_pulso']:.2f} s"
        )

    # Solucoes selecionadas para comparacao formal
    solucoes = {}

    # Solucao A: q'' = 105000 W/m^2, w_laser = 44 mm
    res_A = resolver_placa(h=h, q_flux=105000.0, T_inf=T_inf, N=200, Nt=3000, w_laser=W1)
    met_A = calcular_metricas(res_A)
    solucoes["A"] = {
        "nome": "Solucao A: Ajuste de fluxo (q'' = 105 kW/m^2, w = 44 mm)",
        "curto": "Sol. A (q''=105 kW/m2)",
        "res": res_A,
        "met": met_A,
    }

    # Solucao B: Alargamento do feixe w_laser = 60 mm, q'' = 80000 W/m^2
    res_B = resolver_placa(h=h, q_flux=q_base, T_inf=T_inf, N=200, Nt=3000, w_laser=60.0e-3)
    met_B = calcular_metricas(res_B)
    solucoes["B"] = {
        "nome": "Solucao B: Alargamento do feixe (w_laser = 60 mm, q'' = 80 kW/m^2)",
        "curto": "Sol. B (w_L=60 mm)",
        "res": res_B,
        "met": met_B,
    }

    # Solucao C: Pulso em dois estagios com feixe alargado (w_laser = 60 mm)
    # 0 < t <= 2,0 s: q1 = 180000 W/m^2 (rampa rapida)
    # 2,0 < t <= 10,0 s: q2 = 75000 W/m^2 (patamar de manutencao)
    def pulso_dois_estagios(t_s: float) -> float:
        if t_s <= 2.0 + 1e-12:
            return 180000.0
        if t_s <= 10.0 + 1e-12:
            return 75000.0
        return 0.0

    res_C = resolver_placa(
        h=h, q_flux=pulso_dois_estagios, T_inf=T_inf, N=200, Nt=3000, w_laser=60.0e-3
    )
    met_C = calcular_metricas(res_C)
    solucoes["C"] = {
        "nome": "Solucao C: Pulso 2 estagios + feixe 60 mm (180 -> 75 kW/m^2)",
        "curto": "Sol. C (2 estagios, 60 mm)",
        "res": res_C,
        "met": met_C,
    }

    # Solucao D: Pre-aquecimento da chapa para T_0 = 55 C (q'' = 80 kW/m^2, w = 44 mm)
    res_D = resolver_placa(
        h=h, q_flux=q_base, T_inf=T_inf, N=200, Nt=3000, w_laser=W1, T_init=55.0
    )
    met_D = calcular_metricas(res_D)
    solucoes["D"] = {
        "nome": "Solucao D: Pre-aquecimento da chapa (T_0 = 55 C, q'' = 80 kW/m^2)",
        "curto": "Sol. D (T0=55 C)",
        "res": res_D,
        "met": met_D,
    }

    # Teste complementar: reducao de h para 100 W/(m^2.K) (carenagem/controle de fluxo de ar)
    res_h100 = resolver_placa(h=100.0, q_flux=q_base, T_inf=T_inf, N=200, Nt=3000, w_laser=W1)
    met_h100 = calcular_metricas(res_h100)
    print(
        f"\n[Teste complementar carenagem h=100 W/(m^2.K)]: "
        f"T_max={met_h100['T_max_pulso']:.2f} C, T_borda(10s)={met_h100['T_borda_10s']:.2f} C, "
        f"t_90_borda={met_h100['t_90_borda']:.2f} s, Janela={met_h100['tempo_janela_pulso']:.2f} s"
    )
    solucoes["h100_ref"] = {"res": res_h100, "met": met_h100}

    print("\nResumo comparativo das solucoes selecionadas:")
    for chave in ["A", "B", "C", "D"]:
        m = solucoes[chave]["met"]
        t90b = f"{m['t_90_borda']:.2f} s" if m["t_90_borda"] is not None else "N/A"
        print(
            f"  {solucoes[chave]['nome']}\n"
            f"     T_max={m['T_max_pulso']:.2f} C | T_borda(10s)={m['T_borda_10s']:.2f} C | "
            f"dT_pel={m['delta_T_pel_10s']:.2f} C | t_90_borda={t90b} | "
            f"Janela_pulso={m['tempo_janela_pulso']:.2f} s | Margem_200={m['margem_200']:.2f} C"
        )
    print("=" * 72 + "\n")
    return solucoes


def configurar_estilo_graficos() -> None:
    """Configura padrao visual academico em 300 DPI."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def gerar_figuras(res_base: dict, met_base: dict, solucoes: dict, dir_fig: str) -> None:
    """Gera e salva as 4 figuras do relatorio em formato PDF e PNG (300 DPI)."""
    os.makedirs(dir_fig, exist_ok=True)
    configurar_estilo_graficos()

    x = res_base["x"]
    x_mm = x * 1e3
    t = res_base["t"]
    T = res_base["T"]

    # -------------------------------------------------------------------------
    # Figura 1: Superficie 3D T(x, t) com planos de 90 C e 200 C
    # -------------------------------------------------------------------------
    step_x = max(1, len(x) // 85)
    step_t = max(1, len(t) // 130)
    idx_x = np.unique(np.append(np.arange(0, len(x), step_x), len(x) - 1))
    idx_t = np.unique(np.append(np.arange(0, len(t), step_t), len(t) - 1))
    xs = x[idx_x]
    ts = t[idx_t]
    Ts = T[np.ix_(idx_x, idx_t)]

    TT, XX = np.meshgrid(ts, xs)  # shapes (nx, nt)

    fig = plt.figure(figsize=(8.6, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        XX, TT, Ts, cmap="jet", linewidth=0, antialiased=True, alpha=0.93
    )

    # Planos horizontais em T = 90 C (azul) e T = 200 C (vermelho)
    Xp, Tp = np.meshgrid([0.0, L_MEIO], [0.0, T_FINAL_PADRAO])
    ax.plot_surface(Xp, Tp, np.full_like(Xp, 90.0), color="royalblue", alpha=0.22, shade=False)
    ax.plot_surface(Xp, Tp, np.full_like(Xp, 200.0), color="crimson", alpha=0.20, shade=False)

    ax.set_xlabel("Posição x [m]", labelpad=10)
    ax.set_ylabel("Tempo t [s]", labelpad=10)
    ax.set_zlabel("Temperatura T [°C]", labelpad=8)
    ax.set_title("Grupo 15C: Evolução espaço-temporal T(x, t) na meia chapa", pad=12)
    ax.set_xlim(0.0, L_MEIO)
    ax.set_ylim(0.0, T_FINAL_PADRAO)
    ax.set_zlim(20.0, 220.0)
    ax.view_init(elev=25, azim=-128)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.68, aspect=18, pad=0.08)
    cbar.set_label("Temperatura [°C]")

    patch_90 = mpatches.Patch(color="royalblue", alpha=0.35, label="Limite inferior: 90 °C")
    patch_200 = mpatches.Patch(color="crimson", alpha=0.35, label="Limite superior: 200 °C")
    ax.legend(handles=[patch_90, patch_200], loc="upper left", frameon=True)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(dir_fig, f"fig_superficie_3d.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figura 2: Perfis espaciais T(x) em instantes selecionados
    # -------------------------------------------------------------------------
    instantes = [1.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0]
    cores = ["#4575b4", "#74add1", "#fdae61", "#f46d43", "#d73027", "#5e3c99", "#2c7bb6"]

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.axhspan(90.0, 200.0, xmin=0.0, xmax=22.0 / 66.0, color="mediumseagreen", alpha=0.12,
               label="Janela aceitável na película (90–200 °C)")

    for t_alvo, cor in zip(instantes, cores):
        idx = int(np.argmin(np.abs(t - t_alvo)))
        estilo = "-" if t_alvo <= 10.0 else "--"
        espessura = 2.4 if abs(t_alvo - 10.0) < 1e-6 else 1.7
        t_str = f"{t_alvo:g}".replace(".", ",")
        fase = "aquec." if t_alvo <= 10.0 else "resfr."
        ax.plot(
            x_mm, T[:, idx], linestyle=estilo, linewidth=espessura, color=cor,
            label=f"t = {t_str} s ({fase})"
        )

    ax.axvline(22.0, color="black", linestyle=":", linewidth=1.6, label="Borda da película (x = 22 mm)")
    ax.axhline(90.0, color="royalblue", linestyle="--", linewidth=1.2)
    ax.axhline(200.0, color="crimson", linestyle="--", linewidth=1.2)
    ax.text(46.0, 93.0, "T = 90 °C (mínimo)", color="royalblue", fontsize=8.8)
    ax.text(1.0, 203.5, "T = 200 °C", color="crimson", fontsize=8.8)

    ax.set_xlabel("Posição x [mm]")
    ax.set_ylabel("Temperatura T [°C]")
    ax.set_title("Grupo 15C: Perfis espaciais de temperatura na meia chapa")
    ax.set_xlim(0.0, 66.0)
    ax.set_ylim(20.0, 222.0)
    ax.yaxis.set_major_formatter(FuncFormatter(formatar_virgula))
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(dir_fig, f"fig_perfis_espaciais.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figura 3: Evolucao temporal T(t) em posicoes estrategicas
    # -------------------------------------------------------------------------
    posicoes_mm = [0.0, 11.0, 22.0, 44.0]
    rotulos_pos = [
        "x = 0 mm (centro da película)",
        "x = 11 mm (meio da semipelícula)",
        "x = 22 mm (borda da película)",
        "x = 44 mm (região não irradiada)",
    ]
    cores_pos = ["#d73027", "#f46d43", "#1b7837", "#4575b4"]

    fig, ax = plt.subplots(figsize=(8.2, 4.9))
    ax.axhspan(90.0, 200.0, color="mediumseagreen", alpha=0.12, label="Janela admissível (90–200 °C)")
    ax.axvline(10.0, color="dimgray", linestyle="-.", linewidth=1.4, label="Fim do pulso (t = 10 s)")

    for pos_mm, rot, cor in zip(posicoes_mm, rotulos_pos, cores_pos):
        T_pos = np.array([np.interp(pos_mm * 1e-3, x, T[:, j]) for j in range(len(t))])
        ax.plot(t, T_pos, linewidth=2.0, color=cor, label=rot)

    ax.axhline(90.0, color="royalblue", linestyle="--", linewidth=1.1)
    ax.axhline(200.0, color="crimson", linestyle="--", linewidth=1.1)

    # Anotacoes dos valores em t = 10 s (sem sobreposicao)
    T_c10 = met_base["T_centro_10s"]
    T_b10 = met_base["T_borda_10s"]
    ax.scatter([10.0, 10.0], [T_c10, T_b10], color=["#d73027", "#1b7837"], zorder=5, s=36)
    ax.annotate(
        f"Centro (10 s): {T_c10:.1f} °C".replace(".", ","),
        xy=(10.0, T_c10), xytext=(3.6, 168.0),
        arrowprops=dict(arrowstyle="->", color="#d73027", lw=1.1),
        fontsize=8.8, color="#d73027", fontweight="bold"
    )
    ax.annotate(
        f"Borda (10 s): {T_b10:.1f} °C".replace(".", ","),
        xy=(10.0, T_b10), xytext=(12.2, 54.0),
        arrowprops=dict(arrowstyle="->", color="#1b7837", lw=1.1),
        fontsize=8.8, color="#1b7837", fontweight="bold"
    )

    ax.set_xlabel("Tempo t [s]")
    ax.set_ylabel("Temperatura T [°C]")
    ax.set_title("Grupo 15C: Histórico temporal de temperatura em posições estratégicas")
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(20.0, 215.0)
    ax.yaxis.set_major_formatter(FuncFormatter(formatar_virgula))
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(dir_fig, f"fig_evolucao_temporal.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figura 4: Comparacao entre o caso base e as solucoes de engenharia
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9))

    # Painel esquerdo: perfis espaciais em t = 10 s
    ax1.axhspan(90.0, 200.0, xmin=0.0, xmax=22.0 / 66.0, color="mediumseagreen", alpha=0.12)
    ax1.axvline(22.0, color="black", linestyle=":", linewidth=1.5, label="Borda da película (22 mm)")
    ax1.axhline(90.0, color="royalblue", linestyle="--", linewidth=1.1)
    ax1.axhline(200.0, color="crimson", linestyle="--", linewidth=1.1)

    idx_10 = int(np.argmin(np.abs(t - 10.0)))
    ax1.plot(x_mm, T[:, idx_10], color="dimgray", linewidth=2.0, linestyle="--",
             label="Caso Base (80 kW/m², 44 mm)")

    estilos_sol = [
        ("A", "#e66101", "-", "Sol. A: q''=105 kW/m², 44 mm"),
        ("B", "#2b83ba", "-", "Sol. B: w_L=60 mm, 80 kW/m²"),
        ("C", "#1b7837", "-", "Sol. C: 2 estágios + 60 mm (Rec.)"),
        ("D", "#7b3294", "-.", "Sol. D: Pré-aquec. 55 °C"),
    ]
    for chave, cor, ls, rot in estilos_sol:
        r_s = solucoes[chave]["res"]
        idx_s = int(np.argmin(np.abs(r_s["t"] - 10.0)))
        ax1.plot(r_s["x"] * 1e3, r_s["T"][:, idx_s], color=cor, linestyle=ls, linewidth=2.0, label=rot)

    ax1.set_xlabel("Posição x [mm]")
    ax1.set_ylabel("Temperatura em t = 10 s [°C]")
    ax1.set_title("(a) Perfil espacial ao fim do pulso (t = 10 s)")
    ax1.set_xlim(0.0, 66.0)
    ax1.set_ylim(20.0, 215.0)
    ax1.yaxis.set_major_formatter(FuncFormatter(formatar_virgula))
    ax1.legend(loc="upper right", fontsize=8.0, frameon=True, framealpha=0.95)

    # Painel direito: envoltorio temporal T_max(t) e T_min(t) na pelicula (0 <= t <= 10 s)
    ax2.axhspan(90.0, 200.0, color="mediumseagreen", alpha=0.12, label="Janela admissível (90–200 °C)")
    ax2.axhline(90.0, color="royalblue", linestyle="--", linewidth=1.1)
    ax2.axhline(200.0, color="crimson", linestyle="--", linewidth=1.1)

    mask_p = t <= 10.0 + 1e-12
    ax2.plot(t[mask_p], met_base["T_max_pel_t"][mask_p], color="dimgray", lw=2.0, ls="-",
             label="Base: T_max (centro)")
    ax2.plot(t[mask_p], met_base["T_min_pel_t"][mask_p], color="dimgray", lw=1.8, ls=":",
             label="Base: T_min (borda)")

    for chave, cor, nome_curto in [
        ("B", "#2b83ba", "Sol. B"),
        ("C", "#1b7837", "Sol. C (Rec.)"),
    ]:
        r_s = solucoes[chave]["res"]
        m_s = solucoes[chave]["met"]
        mp = r_s["t"] <= 10.0 + 1e-12
        ax2.plot(r_s["t"][mp], m_s["T_max_pel_t"][mp], color=cor, lw=2.1, ls="-",
                 label=f"{nome_curto}: T_max")
        ax2.plot(r_s["t"][mp], m_s["T_min_pel_t"][mp], color=cor, lw=2.0, ls="--",
                 label=f"{nome_curto}: T_min")

    ax2.set_xlabel("Tempo durante o pulso t [s]")
    ax2.set_ylabel("Temperatura na película [°C]")
    ax2.set_title("(b) Faixa térmica na película ao longo do pulso (0–10 s)")
    ax2.set_xlim(0.0, 10.0)
    ax2.set_ylim(20.0, 215.0)
    ax2.yaxis.set_major_formatter(FuncFormatter(formatar_virgula))
    ax2.legend(loc="upper left", ncol=2, fontsize=7.8, frameon=True, framealpha=0.92)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(dir_fig, f"fig_comparacao_solucoes.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Figuras salvas com sucesso em {dir_fig} (300 DPI, PDF e PNG).")


def main() -> None:
    # 1. Validacao obrigatoria (Grupo 11A)
    bench = validar_benchmark_11A()

    # 2. Caso Base do Grupo 15C: h = 150 W/(m^2.K), q'' = 80000 W/m^2, T_inf = 25 C
    h_15C = 150.0
    q_15C = 80000.0
    Tinf_15C = 25.0

    res_15C = resolver_placa(h=h_15C, q_flux=q_15C, T_inf=Tinf_15C, N=200, Nt=3000)
    met_15C = calcular_metricas(res_15C)

    # Verifica posicao exata do no j = 67 em N = 200
    x_no_67_mm = res_15C["x"][67] * 1e3

    # Temperaturas em instantes de resfriamento
    t_vec = res_15C["t"]
    T_mat = res_15C["T"]
    idx_15s = int(np.argmin(np.abs(t_vec - 15.0)))
    idx_20s = int(np.argmin(np.abs(t_vec - 20.0)))
    idx_30s = int(np.argmin(np.abs(t_vec - 30.0)))

    print("=" * 72)
    print("RESULTADOS DO CASO BASE - GRUPO 15C (N=200, Nt=3000)")
    print("=" * 72)
    print(f"alpha (difusividade)     = {res_15C['alpha']:.6e} m^2/s")
    print(f"Bi_y (transversal)       = {res_15C['Bi_y']:.6e}")
    print(f"Bi (equivalente 1D)      = {res_15C['Bi']:.6f}")
    print(f"q_tilde                  = {res_15C['q_tilde']:.6f}")
    print(f"t_tilde_L (10 s)         = {res_15C['t_tilde_L']:.6f}")
    print(f"t_tilde_final (30 s)     = {res_15C['t_tilde_final']:.6f}")
    print(f"No j=67 em x             = {x_no_67_mm:.4f} mm (borda exata = 22,0000 mm)")
    print(f"T_max na pelicula (pulso)= {met_15C['T_max_pulso']:.4f} C em t = {met_15C['t_Tmax']:.2f} s")
    print(f"T(0, 10 s) [centro]      = {met_15C['T_centro_10s']:.4f} C")
    print(f"T(11 mm, 10 s)           = {met_15C['T_11mm_10s']:.4f} C")
    print(f"T(22 mm, 10 s) [borda]   = {met_15C['T_borda_10s']:.4f} C")
    print(f"T(44 mm, 10 s)           = {met_15C['T_44mm_10s']:.4f} C")
    print(f"Gradiente na pelicula    = {met_15C['delta_T_pel_10s']:.4f} C")
    print(f"t_90_centro              = {met_15C['t_90_centro']:.4f} s")
    print(f"t_90_borda               = {met_15C['t_90_borda']}")
    print(f"Tempo janela [90,200]    = {met_15C['tempo_janela_pulso']:.4f} s (pulso) | {met_15C['tempo_janela_total']:.4f} s (total)")
    print(f"Violacao T > 200 C       = {met_15C['violou_200']}")
    print(f"Resfriamento T(0, 15 s)  = {T_mat[0, idx_15s]:.4f} C | T(22 mm, 15 s) = {np.interp(22e-3, res_15C['x'], T_mat[:, idx_15s]):.4f} C")
    print(f"Resfriamento T(0, 20 s)  = {T_mat[0, idx_20s]:.4f} C | T(22 mm, 20 s) = {np.interp(22e-3, res_15C['x'], T_mat[:, idx_20s]):.4f} C")
    print(f"Resfriamento T(0, 30 s)  = {T_mat[0, idx_30s]:.4f} C | T(22 mm, 30 s) = {np.interp(22e-3, res_15C['x'], T_mat[:, idx_30s]):.4f} C")
    print("=" * 72 + "\n")

    # 3. Convergencia de malha
    conv = estudo_convergencia_malha(h=h_15C, q_flux=q_15C, T_inf=Tinf_15C)

    # 4. Solucoes de engenharia
    solucoes = simular_solucoes_engenharia(h=h_15C, q_base=q_15C, T_inf=Tinf_15C)

    # 5. Geracao de figuras em overleaf/figuras/
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dir_fig = os.path.join(raiz, "overleaf", "figuras")
    gerar_figuras(res_15C, met_15C, solucoes, dir_fig)

    # Exportacao de resultados exatos em JSON para auditoria e LaTeX
    def resumir_sol(s_dict):
        m = s_dict["met"]
        return {
            "T_max_pulso": round(m["T_max_pulso"], 4),
            "t_Tmax": round(m["t_Tmax"], 4),
            "T_centro_10s": round(m["T_centro_10s"], 4),
            "T_borda_10s": round(m["T_borda_10s"], 4),
            "delta_T_pel_10s": round(m["delta_T_pel_10s"], 4),
            "t_90_centro": round(m["t_90_centro"], 4) if m["t_90_centro"] is not None else None,
            "t_90_borda": round(m["t_90_borda"], 4) if m["t_90_borda"] is not None else None,
            "tempo_janela_pulso": round(m["tempo_janela_pulso"], 4),
            "tempo_janela_total": round(m["tempo_janela_total"], 4),
            "margem_200": round(m["margem_200"], 4),
            "violou_200": m["violou_200"],
        }

    # Limites exatos de q'' para Solucao A (linearidade em T - T_inf)
    dT_max_por_W = (met_15C["T_max_pulso"] - Tinf_15C) / q_15C
    dT_borda_por_W = (met_15C["T_borda_10s"] - Tinf_15C) / q_15C
    q_min_90 = (90.0 - Tinf_15C) / dT_borda_por_W
    q_max_200 = (200.0 - Tinf_15C) / dT_max_por_W
    print(f"Limites teoricos Solucao A (w_laser=44 mm): q_min(90C)={q_min_90:.1f} W/m^2, q_max(200C)={q_max_200:.1f} W/m^2")

    dados_saida = {
        "benchmark_11A": {
            "Bi": round(bench["res"]["Bi"], 6),
            "q_tilde": round(bench["res"]["q_tilde"], 6),
            "T_max_pulso": round(bench["met"]["T_max_pulso"], 4),
            "t_90_borda": round(bench["met"]["t_90_borda"], 4),
            "T_borda_10s": round(bench["met"]["T_borda_10s"], 4),
        },
        "grupo_15C_base": {
            "alpha": res_15C["alpha"],
            "Bi_y": res_15C["Bi_y"],
            "Bi": round(res_15C["Bi"], 6),
            "q_tilde": round(res_15C["q_tilde"], 6),
            "t_tilde_L": round(res_15C["t_tilde_L"], 6),
            "t_tilde_final": round(res_15C["t_tilde_final"], 6),
            "T_max_pulso": round(met_15C["T_max_pulso"], 4),
            "T_centro_10s": round(met_15C["T_centro_10s"], 4),
            "T_11mm_10s": round(met_15C["T_11mm_10s"], 4),
            "T_borda_10s": round(met_15C["T_borda_10s"], 4),
            "T_44mm_10s": round(met_15C["T_44mm_10s"], 4),
            "delta_T_pel_10s": round(met_15C["delta_T_pel_10s"], 4),
            "t_90_centro": round(met_15C["t_90_centro"], 4),
            "t_90_borda": met_15C["t_90_borda"],
            "tempo_janela_pulso": round(met_15C["tempo_janela_pulso"], 4),
            "tempo_janela_total": round(met_15C["tempo_janela_total"], 4),
            "violou_200": met_15C["violou_200"],
            "T_centro_15s": round(float(T_mat[0, idx_15s]), 4),
            "T_borda_15s": round(float(np.interp(22e-3, res_15C["x"], T_mat[:, idx_15s])), 4),
            "T_centro_20s": round(float(T_mat[0, idx_20s]), 4),
            "T_borda_20s": round(float(np.interp(22e-3, res_15C["x"], T_mat[:, idx_20s])), 4),
            "T_centro_30s": round(float(T_mat[0, idx_30s]), 4),
            "T_borda_30s": round(float(np.interp(22e-3, res_15C["x"], T_mat[:, idx_30s])), 4),
        },
        "convergencia": conv,
        "solucoes": {
            "limites_A": {"q_min_90": round(q_min_90, 1), "q_max_200": round(q_max_200, 1)},
            "A": resumir_sol(solucoes["A"]),
            "B": resumir_sol(solucoes["B"]),
            "C": resumir_sol(solucoes["C"]),
            "D": resumir_sol(solucoes["D"]),
            "h100_ref": resumir_sol(solucoes["h100_ref"]),
        },
    }

    caminho_json = os.path.join(raiz, "src", "resultados_15C.json")
    with open(caminho_json, "w", encoding="utf-8") as arq:
        json.dump(dados_saida, arq, indent=2, ensure_ascii=False)
    print(f"Resultados exportados para {caminho_json}")


if __name__ == "__main__":
    main()
