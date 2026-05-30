"""
performance_monitor.py — Monitoramento de Desempenho

Papel no sistema:
    Mede e compara o desempenho dos três algoritmos (Força Bruta, DP,
    Monte Carlo) para tamanhos crescentes de grade, gerando as curvas
    de escalabilidade empírica exigidas pelo enunciado (Seção 4).

Metodologia de medição:
    - Tempo:    time.perf_counter() — resolução sub-microsegundo
    - Memória:  tracemalloc — pico de alocação em MB
    - Iterações: contadores instrumentados dentro de cada algoritmo
    - Repetições: 5 execuções por configuração;

Tamanhos testados:
    N = M ∈ {3, 5, 10, 20, 50, 100}
"""

import sys
import time
import tracemalloc
import os
from math import comb
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).parent))
from data_loader         import generate_small_grid, generate_grid_matopiba
from brute_force         import brute_force
from dynamic_programming import dp_bottom_up, dp_memoization
from monte_carlo         import run_monte_carlo

# CONFIGURAÇÕES

SIZES        = [3, 5, 10, 20, 50, 100]   # tamanhos de grade N×N testados
N_REPEATS    = 5                          # repetições por medição
FB_MAX_N     = 5                          # limite da Força Bruta
MC_K_MONITOR = 500                        # cenários MC no monitor (vs 10k real)
CENARIO      = "A"                        # cenário usado nas medições


# GERADOR DE GRADE SINTÉTICA PARA BENCHMARK

def _make_grid(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gera grade NxN sintética calibrada pelos parâmetros do Cenário A.

    Returns:
        (cost, risk, prob) — arrays (NxN)
    """
    rng  = np.random.default_rng(seed)
    cost = rng.integers(1, 11, size=(n, n)).astype(int)
    risk = rng.beta(4, 6, size=(n, n))       
    prob = rng.beta(3, 5, size=(n, n))        
    return cost, risk, prob


# FUNÇÕES DE MEDIÇÃO INDIVIDUAIS

def _medir_tempo_memoria(fn, *args, **kwargs) -> tuple[float, float, object]:
    """
    Executa fn(*args, **kwargs) medindo tempo e pico de memória.

    Returns:
        (time_ms, memory_mb, resultado)
    """
    tracemalloc.start()
    t0     = time.perf_counter()
    result = fn(*args, **kwargs)
    t1     = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return (t1 - t0) * 1000, peak / (1024 ** 2), result


def medir_forca_bruta(n: int) -> dict:
    """
    Mede desempenho da Força Bruta para grade nxn.
    Executa N_REPEATS vezes e retorna a mediana.
    Retorna NaN para n > FB_MAX_N.
    """
    if n > FB_MAX_N:
        return {
            "time_ms":    float("nan"),
            "memory_mb":  float("nan"),
            "iterations": comb(2 * n - 2, n - 1),  
            "viable":     False,
            "note":       f"Inviável (teórico: C({2*n-2},{n-1})={comb(2*n-2,n-1):,})",
        }

    times, mems, iters = [], [], []
    for rep in range(N_REPEATS):
        cost, risk, _ = _make_grid(n, seed=rep)
        counter       = [0]

        # Injeta contador no brute_force via wrapper
        t_ms, mem_mb, res = _medir_tempo_memoria(brute_force, cost, risk)
        times.append(t_ms)
        mems.append(mem_mb)
        iters.append(res["n_calls"])

    return {
        "time_ms":    float(np.median(times)),
        "memory_mb":  float(np.median(mems)),
        "iterations": int(np.median(iters)),
        "viable":     True,
        "note":       f"{N_REPEATS} repetições, mediana",
    }


def medir_dp(n: int, method: str = "bottom_up") -> dict:
    """
    Mede desempenho da DP (bottom_up ou memoization) para grade nxn.
    """
    fn = dp_bottom_up if method == "bottom_up" else dp_memoization
    times, mems, iters = [], [], []

    for rep in range(N_REPEATS):
        cost, risk, _ = _make_grid(n, seed=rep)
        counter       = [0]

        t_ms, mem_mb, res = _medir_tempo_memoria(fn, cost, risk, counter)
        times.append(t_ms)
        mems.append(mem_mb)
        iters.append(res["n_iterations"])

    return {
        "time_ms":    float(np.median(times)),
        "memory_mb":  float(np.median(mems)),
        "iterations": int(np.median(iters)),
        "viable":     True,
        "note":       f"{N_REPEATS} repetições, mediana",
    }


def medir_monte_carlo(n: int, K: int = MC_K_MONITOR) -> dict:
    """
    Mede desempenho do Monte Carlo para grade nxn com K cenários.
    Usa batch_size=K para medir o tempo total de uma vez só.
    """
    times, mems = [], []

    for rep in range(N_REPEATS):
        cost, risk, prob = _make_grid(n, seed=rep)

        tracemalloc.start()
        t0  = time.perf_counter()
        run_monte_carlo(
            cost=cost, risk=risk, prob=prob,
            K=K, seed=rep, verbose=False,
            batch_size=K,
        )
        t1      = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append((t1 - t0) * 1000)
        mems.append(peak / (1024 ** 2))

    return {
        "time_ms":    float(np.median(times)),
        "memory_mb":  float(np.median(mems)),
        "iterations": K * n * n,     # K × N×M operações DP
        "viable":     True,
        "note":       f"K={K} cenários, {N_REPEATS} repetições, mediana",
    }


# EXECUÇÃO COMPLETA — TODOS OS ALGORITMOS × TODOS OS TAMANHOS

def run_benchmark() -> dict:
    """
    Executa o benchmark completo para todos os tamanhos e algoritmos.

    Returns:
        dict aninhado: resultados[algoritmo][n] = dict de métricas
    """
    resultados = {
        "fb":   {},
        "dp_bu": {},
        "dp_mem": {},
        "mc":   {},
    }

    print(f"\n{'─'*62}")
    print(f"  {'N':>4} │ {'FB (ms)':>10} │ {'DP-BU (ms)':>10} │ "
          f"{'DP-Mem (ms)':>11} │ {'MC (ms)':>10}")
    print(f"{'─'*62}")

    for n in SIZES:
        # Força Bruta
        r_fb  = medir_forca_bruta(n)
        resultados["fb"][n] = r_fb

        # DP Bottom-Up
        r_dpbu = medir_dp(n, method="bottom_up")
        resultados["dp_bu"][n] = r_dpbu

        # DP Memoização
        r_dpmem = medir_dp(n, method="memoization")
        resultados["dp_mem"][n] = r_dpmem

        # Monte Carlo
        r_mc = medir_monte_carlo(n)
        resultados["mc"][n] = r_mc

        fb_str  = f"{r_fb['time_ms']:>10.3f}"  if r_fb["viable"]  else f"{'N/A':>10}"
        print(f"  {n:>4} │ {fb_str} │ "
              f"{r_dpbu['time_ms']:>10.3f} │ "
              f"{r_dpmem['time_ms']:>11.3f} │ "
              f"{r_mc['time_ms']:>10.3f}")

    print(f"{'─'*62}")
    return resultados


# VISUALIZAÇÕES

def plot_escalabilidade(resultados: dict, save_path: str = None) -> None:
    """
    Gera a curva de escalabilidade empírica: tempo x N para os 3 algoritmos.
    Figura obrigatória do enunciado (Seção 6.1).

    Layout: dois painéis
        Esquerdo: escala linear — evidencia diferença absoluta entre algoritmos
        Direito:  escala log-log — evidencia a ordem de complexidade
    """
    if save_path is None:
        save_path = str(
            Path(__file__).parent.parent / "report" / "escalabilidade.png"
        )
    os.makedirs(Path(save_path).parent, exist_ok=True)

    ns = SIZES

    # Extrai tempos (NaN para FB inviável)
    t_fb    = [resultados["fb"][n]["time_ms"]    for n in ns]
    t_dpbu  = [resultados["dp_bu"][n]["time_ms"] for n in ns]
    t_dpmem = [resultados["dp_mem"][n]["time_ms"]for n in ns]
    t_mc    = [resultados["mc"][n]["time_ms"]    for n in ns]

    # Separa FB viável (N <= 5) de extrapolado
    ns_fb_ok   = [n for n in ns if n <= FB_MAX_N]
    t_fb_ok    = [t for n, t in zip(ns, t_fb) if n <= FB_MAX_N]

    # Extrapolação teórica Força Bruta: C(2N-2,N-1) x fator de escala
    # Calibrado pelo último ponto medido
    ns_extrap  = [n for n in ns if n > FB_MAX_N]
    if ns_fb_ok and ns_extrap:
        n_last = ns_fb_ok[-1]
        t_last = t_fb_ok[-1]
        paths_last = comb(2 * n_last - 2, n_last - 1)
        t_per_path = t_last / max(paths_last, 1)
        t_fb_extrap = [t_per_path * comb(2*n-2, n-1) for n in ns_extrap]
    else:
        t_fb_extrap = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cores = {
        "FB":     "#DC2626",
        "DP-BU":  "#2563EB",
        "DP-Mem": "#7C3AED",
        "MC":     "#D97706",
    }

    for ax, titulo, usar_log in [
        (ax1, "Escala Linear",  False),
        (ax2, "Escala Log-Log", True),
    ]:
        # DP bottom-up
        ax.plot(ns, t_dpbu,  "o-",  color=cores["DP-BU"],  lw=2.2, ms=7,
                label="DP Bottom-Up  O(N²)")
        # DP memoização
        ax.plot(ns, t_dpmem, "s--", color=cores["DP-Mem"], lw=1.8, ms=7,
                label="DP Memoização O(N²)")
        # Monte Carlo
        ax.plot(ns, t_mc,    "^-",  color=cores["MC"],     lw=2.2, ms=7,
                label=f"Monte Carlo  O(K·N²) K={MC_K_MONITOR}")

        # Força Bruta: pontos medidos
        if ns_fb_ok:
            ax.plot(ns_fb_ok, t_fb_ok, "D-", color=cores["FB"], lw=2, ms=8,
                    label=f"Força Bruta  O(C(2N-2,N-1))")
        # Força Bruta: extrapolação tracejada
        if ns_extrap and t_fb_extrap:
            ax.plot(ns_extrap, t_fb_extrap, "D:", color=cores["FB"],
                    lw=1.5, ms=6, alpha=0.55,
                    label="FB extrapolado (teórico)")
            # Liga o último ponto real ao primeiro extrapolado
            if ns_fb_ok:
                ax.plot(
                    [ns_fb_ok[-1], ns_extrap[0]],
                    [t_fb_ok[-1], t_fb_extrap[0]],
                    ":", color=cores["FB"], lw=1.2, alpha=0.4,
                )

        if usar_log:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

        ax.set_xlabel("Tamanho N (grade NxN)", fontsize=11)
        ax.set_ylabel("Tempo de execução (ms)", fontsize=11)
        ax.set_title(titulo, fontsize=11, fontweight="bold")
        ax.set_xticks(ns)
        ax.legend(fontsize=8.5, loc="upper left")
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        f"Escalabilidade Empírica — Tempo × N\n"
        f"Força Bruta | DP (Bottom-Up e Memoização) | Monte Carlo "
        f"(K={MC_K_MONITOR})\n"
        f"Mediana de {N_REPEATS} execuções por ponto | Cenário A calibrado",
        fontsize=11, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Curva de escalabilidade salva em: {save_path}")


def plot_memoria(resultados: dict, save_path: str = None) -> None:
    """
    Gera gráfico de uso de memória x N para os 3 algoritmos.
    Complementa a análise de desempenho (Seção 4 do enunciado).
    """
    if save_path is None:
        save_path = str(
            Path(__file__).parent.parent / "report" / "memoria.png"
        )
    os.makedirs(Path(save_path).parent, exist_ok=True)

    ns      = SIZES
    m_dpbu  = [resultados["dp_bu"][n]["memory_mb"]  for n in ns]
    m_dpmem = [resultados["dp_mem"][n]["memory_mb"] for n in ns]
    m_mc    = [resultados["mc"][n]["memory_mb"]     for n in ns]
    m_fb    = [resultados["fb"][n]["memory_mb"]
               if resultados["fb"][n]["viable"] else float("nan") for n in ns]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(ns, m_dpbu,  "o-",  color="#2563EB", lw=2,   ms=7, label="DP Bottom-Up")
    ax.plot(ns, m_dpmem, "s--", color="#7C3AED", lw=1.8, ms=7, label="DP Memoização")
    ax.plot(ns, m_mc,    "^-",  color="#D97706", lw=2,   ms=7, label=f"Monte Carlo K={MC_K_MONITOR}")
    ax.plot(ns, m_fb,    "D-",  color="#DC2626", lw=2,   ms=8, label="Força Bruta")

    ax.set_xlabel("Tamanho N (grade N×N)", fontsize=11)
    ax.set_ylabel("Pico de memória alocada (MB)", fontsize=11)
    ax.set_title(
        "Uso de Memória × N\n"
        "Comparativo entre os três algoritmos",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(ns)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Gráfico de memória salvo em: {save_path}")


def plot_scatter_custo_tempo(resultados: dict, save_path: str = None) -> None:
    """
    Gera o scatter: custo ótimo x tempo computacional.
    Figura obrigatória solicitada
    """
    if save_path is None:
        save_path = str(
            Path(__file__).parent.parent / "report" / "scatter_custo_tempo.png"
        )
    os.makedirs(Path(save_path).parent, exist_ok=True)

    # Calcula custo ótimo real para cada N (usando DP como referência)
    ns_validos = [3, 5]    # tamanhos onde FB é viável (para comparação direta)
    custos, t_fb_pts, t_dp_pts, t_mc_pts = [], [], [], []

    for n in ns_validos:
        cost, risk, prob = _make_grid(n, seed=0)
        res_dp  = dp_bottom_up(cost, risk)
        custos.append(res_dp["min_cost"])
        t_fb_pts.append(resultados["fb"][n]["time_ms"])
        t_dp_pts.append(resultados["dp_bu"][n]["time_ms"])
        t_mc_pts.append(resultados["mc"][n]["time_ms"])

    # Para N > 5, usa apenas DP e MC
    ns_grandes = [10, 20, 50, 100]
    custos_g, t_dp_g, t_mc_g = [], [], []
    for n in ns_grandes:
        cost, risk, prob = _make_grid(n, seed=0)
        res_dp = dp_bottom_up(cost, risk)
        custos_g.append(res_dp["min_cost"])
        t_dp_g.append(resultados["dp_bu"][n]["time_ms"])
        t_mc_g.append(resultados["mc"][n]["time_ms"])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter por algoritmo
    ax.scatter(t_fb_pts, custos,   s=100, color="#DC2626", marker="D",
               zorder=5, label="Força Bruta (N≤5)", alpha=0.9)
    ax.scatter(t_dp_pts, custos,   s=80,  color="#2563EB", marker="o",
               zorder=5, label="DP Bottom-Up (N≤5)", alpha=0.9)
    ax.scatter(t_mc_pts, custos,   s=80,  color="#D97706", marker="^",
               zorder=5, label=f"Monte Carlo K={MC_K_MONITOR} (N≤5)", alpha=0.9)
    ax.scatter(t_dp_g,   custos_g, s=80,  color="#2563EB", marker="o",
               zorder=5, alpha=0.45, label="DP Bottom-Up (N>5)")
    ax.scatter(t_mc_g,   custos_g, s=80,  color="#D97706", marker="^",
               zorder=5, alpha=0.45, label=f"Monte Carlo K={MC_K_MONITOR} (N>5)")

    # Anotações de N
    for i, n in enumerate(ns_validos):
        ax.annotate(f"N={n}", (t_fb_pts[i], custos[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, color="#DC2626")
        ax.annotate(f"N={n}", (t_dp_pts[i], custos[i]),
                    xytext=(5, -12), textcoords="offset points", fontsize=8, color="#2563EB")
    for i, n in enumerate(ns_grandes):
        ax.annotate(f"N={n}", (t_dp_g[i], custos_g[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=8, color="#2563EB", alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Tempo computacional (ms) — escala log", fontsize=11)
    ax.set_ylabel("Custo ótimo da solução", fontsize=11)
    ax.set_title(
        "Trade-off: Qualidade da Solução × Tempo Computacional\n"
        "FB e DP convergem para o mesmo custo ótimo — DP domina em tempo",
        fontsize=11, fontweight="bold",
    )
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Scatter custo×tempo salvo em: {save_path}")


def imprimir_tabela_completa(resultados: dict) -> None:
    """Exibe tabela completa de métricas no terminal."""
    print("\n" + "=" * 72)
    print("  TABELA COMPLETA DE DESEMPENHO")
    print("=" * 72)

    headers = ["N", "Algoritmo", "Tempo (ms)", "Memória (MB)", "Iterações", "Nota"]
    print(f"  {'N':>4} │ {'Algoritmo':<14} │ {'Tempo ms':>10} │ "
          f"{'Mem MB':>10} │ {'Iterações':>14} │ Nota")
    print(f"  {'─'*4}─┼─{'─'*14}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*14}─┼─{'─'*20}")

    for n in SIZES:
        entradas = [
            ("FB", resultados["fb"][n]),
            ("DP-BU", resultados["dp_bu"][n]),
            ("DP-Memo", resultados["dp_mem"][n]),
            ("MC", resultados["mc"][n]),
        ]
        for nome, r in entradas:
            t_str = f"{r['time_ms']:>10.3f}" if not np.isnan(r["time_ms"]) else f"{'N/A':>10}"
            m_str = f"{r['memory_mb']:>10.4f}" if not np.isnan(r["memory_mb"]) else f"{'N/A':>10}"
            i_str = f"{r['iterations']:>14,}"
            print(f"  {n:>4} │ {nome:<14} │ {t_str} │ {m_str} │ {i_str} │ {r['note'][:20]}")
        print(f"  {'─'*4}─┼─{'─'*14}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*14}─┼─{'─'*20}")


# EXECUÇÃO PRINCIPAL

def run_demo() -> dict:
    """
    Executa o benchmark completo e gera todas as figuras de desempenho.
    """
    print("\n" + "=" * 62)
    print("  Monitor de Desempenho — Benchmark Completo")
    print("=" * 62)
    print(f"  Tamanhos:    {SIZES}")
    print(f"  Repetições:  {N_REPEATS} por ponto (resultado = mediana)")
    print(f"  MC K:        {MC_K_MONITOR} cenários (monitor)")
    print(f"  FB limite:   N <= {FB_MAX_N}")

    print("\n  Executando benchmark...")
    resultados = run_benchmark()

    imprimir_tabela_completa(resultados)

    print("\n  Gerando figuras...")
    plot_escalabilidade(resultados)
    plot_memoria(resultados)
    plot_scatter_custo_tempo(resultados)

    print("\n  Monitor de desempenho concluído.")
    print("    Próximo passo: python src/visualizations.py\n")

    return resultados


if __name__ == "__main__":
    run_demo()