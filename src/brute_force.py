"""
brute_force.py — Enumeração Completa (Força Bruta)

Baseline de validação para instâncias pequenas (N, M ≤ 5).
Serve como oráculo: para toda instância pequena, o resultado deve coincidir com o da Programação Dinâmica (Etapa 4).

Complexidade:
    Número de caminhos = C(N+M-2, N-1) — combinatório.
    Tempo: O(C(N+M-2, N-1)) — exponencial em N e M.
    Espaço: O(N+M).
"""

import sys
import time
import tracemalloc
from math import comb
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import generate_small_grid


# CONSTANTES

MAX_N = 5   # tamanho máximo permitido para Força Bruta
INF   = float("inf")


# NÚCLEO — ENUMERAÇÃO RECURSIVA

def _enumerate_paths(
    cost:    np.ndarray,
    risk:    np.ndarray,
    i:       int,
    j:       int,
    n:       int,
    m:       int,
    path:    list,
    counter: list,          
    all_paths: list,        
    curr_cost: float,
) -> None:
    """
    Percorre recursivamente todos os caminhos de (i,j) até (n-1, m-1).
    Movimentos permitidos: apenas direita (+j) ou baixo (+i).
    """
    counter[0] += 1

    # Célula bloqueada: abandona este ramo
    if cost[i, j] == -1:
        return

    # Custo da célula atual: c[i][j] × (1 + r[i][j])
    # O fator (1 + r[i][j]) pondera o risco como multiplicador do custo base
    cell_cost = cost[i, j] * (1.0 + risk[i, j])
    total     = curr_cost + cell_cost
    path      = path + [(i, j)]   # cria nova lista — não modifica o ramo pai

    if i == n - 1 and j == m - 1:
        all_paths.append((total, path))
        return

    # Movimento para baixo
    if i + 1 < n:
        _enumerate_paths(cost, risk, i + 1, j, n, m, path, counter, all_paths, total)

    # Movimento para a direita
    if j + 1 < m:
        _enumerate_paths(cost, risk, i, j + 1, n, m, path, counter, all_paths, total)


def brute_force(
    cost: np.ndarray,
    risk: np.ndarray,
) -> dict:
    """
    Encontra o caminho de custo mínimo de (0,0) a (N-1, M-1)
    por enumeração completa de todos os caminhos válidos.

    Fórmula de custo de cada célula:
        custo_celula(i,j) = cost[i][j] x (1 + risk[i][j])
    Custo total do caminho = soma dos custos de todas as células visitadas.

    Restrições:
        - Movimentos: apenas direita ou baixo
        - Células com cost[i][j] == -1 são intransponíveis (bloqueadas)
        - A grade deve ter N, M ≤ MAX_N (5) para ser viável
    """
    n, m = cost.shape

    if n > MAX_N or m > MAX_N:
        raise ValueError(
            f"Força Bruta limitada a grades {MAX_N}x{MAX_N}. "
            f"Recebeu {n}x{m}. Use Programação Dinâmica para grades maiores."
        )

    # Instrumentação: memória e tempo
    tracemalloc.start()
    t0 = time.perf_counter()

    counter   = [0]   # contador de chamadas (lista para ser mutável na recursão)
    all_paths = []    # (custo_total, caminho) de cada caminho válido

    _enumerate_paths(
        cost=cost, risk=risk,
        i=0, j=0, n=n, m=m,
        path=[], counter=counter,
        all_paths=all_paths,
        curr_cost=0.0,
    )

    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    time_ms = (t1 - t0) * 1000
    memory_mb = peak / (1024 ** 2)

    if not all_paths:
        return {
            "min_cost":     INF,
            "optimal_path": [],
            "all_paths":    [],
            "n_paths":      0,
            "n_calls":      counter[0],
            "time_ms":      time_ms,
            "memory_mb":    memory_mb,
            "feasible":     False,
        }

    # Seleciona o caminho de menor custo
    all_paths.sort(key=lambda x: x[0])
    min_cost, optimal_path = all_paths[0]

    return {
        "min_cost":     min_cost,
        "optimal_path": optimal_path,
        "all_paths":    all_paths,
        "n_paths":      len(all_paths),
        "n_calls":      counter[0],
        "time_ms":      time_ms,
        "memory_mb":    memory_mb,
        "feasible":     True,
    }


# GRÁFICO: CRESCIMENTO EXPONENCIAL DO Nº DE CAMINHOS

def plot_path_growth(save_path: str = None) -> None:
    """
    Gera o gráfico do crescimento do número de caminhos em função de N
    (para grades NxN quadradas), evidenciando o comportamento combinatório.

    O número exato de caminhos numa grade NxN é:
        C(2N-2, N-1) = (2N-2)! / ((N-1)! x (N-1)!)

    O gráfico compara:
        - Nº real de caminhos C(2N-2, N-1) — escala log
        - Nº de chamadas recursivas medido empiricamente
        - Linha de referência 2^N — para comparação visual
    """
    if save_path is None:
        save_path = str(Path(__file__).parent.parent / "report" / "fb_crescimento_caminhos.png")
    os.makedirs(Path(save_path).parent, exist_ok=True)

    ns = list(range(2, MAX_N + 1))          
    n_paths_teo = [comb(2*n - 2, n - 1) for n in ns] # C(2N-2, N-1) teórico
    n_calls_emp = []                                   

    # Grade sintética sem bloqueios para medir chamadas puras
    rng = np.random.default_rng(0)
    for n in ns:
        cost_test = rng.integers(1, 6, size=(n, n))
        risk_test = rng.random(size=(n, n))
        res       = brute_force(cost_test, risk_test)
        n_calls_emp.append(res["n_calls"])

    ref_2n = [2 ** n for n in ns]   

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.semilogy(ns, n_paths_teo, "o-",  color="#2563EB", lw=2,   ms=8,  label="Caminhos válidos C(2N-2, N-1)")
    ax.semilogy(ns, n_calls_emp, "s--", color="#DC2626", lw=1.5, ms=7,  label="Chamadas recursivas (medido)")
    ax.semilogy(ns, ref_2n,      "^:", color="#6B7280",  lw=1,   ms=6,  label="Referência 2ᴺ")

    # Anotações nos pontos
    for n, np_, nc in zip(ns, n_paths_teo, n_calls_emp):
        ax.annotate(f"{np_}", xy=(n, np_), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color="#2563EB")

    ax.set_xlabel("Tamanho N (grade NxN)", fontsize=11)
    ax.set_ylabel("Quantidade (escala logarítmica)", fontsize=11)
    ax.set_title(
        "Força Bruta — Crescimento do Número de Caminhos\n"
        "Grade NxN, movimentos: direita e baixo apenas",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks(ns)

    # Caixa de texto com valores exatos
    tabela = "\n".join(
        [f"N={n}: {c:>6} caminhos | {k:>6} chamadas"
         for n, c, k in zip(ns, n_paths_teo, n_calls_emp)]
    )
    ax.text(0.02, 0.97, tabela, transform=ax.transAxes,
            fontsize=7.5, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Gráfico salvo em: {save_path}")


# DEMONSTRAÇÃO NOS CENÁRIOS REAIS (grades 4×4 e 5×5)

def _print_grid(cost: np.ndarray, risk: np.ndarray, path: list, label: str) -> None:
    """Exibe a grade no terminal com o caminho ótimo marcado com '*'."""
    n, m = cost.shape
    path_set = set(path)
    print(f"\n  {label}  (custo | risco)")
    print("  " + "─" * (m * 12))
    for i in range(n):
        row = ""
        for j in range(m):
            cell = f"c={cost[i,j]:2d} r={risk[i,j]:.2f}"
            if cost[i, j] == -1:
                cell = "  BLOQ.    "
            marker = "★" if (i, j) in path_set else " "
            row += f" {marker}{cell} |"
        print(f"  {row}")
    print("  " + "─" * (m * 12))


def run_demo() -> None:
    """
    Executa a Força Bruta nos dois cenários com grades pequenas,
    exibe os resultados e gera o gráfico de crescimento.
    """
    import os
    print("\n" + "=" * 58)
    print("  Força Bruta — Demonstração nos Cenários A e C")
    print("=" * 58)

    configs = [
        ("A", 4, 4, "Cenário A — MATOPIBA (4×4)"),
        ("A", 5, 5, "Cenário A — MATOPIBA (5×5)"),
        ("C", 4, 4, "Cenário C — Amazônia (4×4)"),
        ("C", 5, 5, "Cenário C — Amazônia (5×5)"),
    ]

    for cenario, n, m, label in configs:
        grid = generate_small_grid(n=n, m=m, cenario=cenario, seed=7)
        cost = grid["cost"]
        risk = grid["risk"]

        print(f"\n{'─'*58}")
        print(f"  {label}")
        print(f"{'─'*58}")

        res = brute_force(cost, risk)

        _print_grid(cost, risk, res["optimal_path"], label)

        if res["feasible"]:
            print(f"\n  Custo mínimo:       {res['min_cost']:.4f}")
            print(f"  Caminho ótimo:      {res['optimal_path']}")
            print(f"  Caminhos válidos:   {res['n_paths']}")
            print(f"  Chamadas recursivas:{res['n_calls']}")
            print(f"  Tempo:              {res['time_ms']:.3f} ms")
            print(f"  Memória (pico):     {res['memory_mb']:.4f} MB")
            print(f"  Teórico C({2*n-2},{n-1}):  {comb(2*n-2, n-1)} caminhos")
        else:
            print("  Nenhum caminho viável encontrado (grade muito bloqueada).")

    print(f"\n{'─'*58}")
    print("  Gerando gráfico de crescimento exponencial...")
    plot_path_growth()
    print("  Força Bruta concluída.")


# EXECUÇÃO DIRETA

import os

if __name__ == "__main__":
    run_demo()