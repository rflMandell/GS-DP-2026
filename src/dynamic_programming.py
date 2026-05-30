"""
dynamic_programming.py — Programação Dinâmica 2D

Complexidade:
    Tabulação bottom-up:  O(NxM) tempo | O(NxM) espaço
    Memoização top-down:  O(NxM) tempo | O(NxM) espaço + overhead de pilha
    Reconstrução:         O(N+M) tempo | O(N+M) espaço
"""

import sys
import time
import tracemalloc
import functools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import generate_small_grid, load_grid, GRID_SIZE

INF = float("inf")


# FUNÇÕES AUXILIARES

def _cell_weight(cost: np.ndarray, risk: np.ndarray, i: int, j: int) -> float:
    """
    Calcula o peso efetivo da célula (i,j).

    Fórmula: w(i,j) = cost[i][j] x (1 + risk[i][j])
    O fator (1 + risk) pondera o risco como multiplicador do custo logístico:
    risco zero → custo base; risco máximo (1.0) → custo dobrado.
    """
    if cost[i, j] == -1:
        return INF
    return float(cost[i, j]) * (1.0 + float(risk[i, j]))


def reconstruct_path(dp: np.ndarray, cost: np.ndarray) -> list[tuple[int, int]]:
    """
    Reconstrói o caminho ótimo por backtracking sobre a tabela DP.

    Parte do destino (N-1, M-1) e volta à origem (0,0) escolhendo sempre
    o vizinho (cima ou esquerda) com menor valor em dp. Inverte ao final.
    """
    n, m = dp.shape

    if dp[n - 1, m - 1] == INF:
        return []   # destino inacessível

    path = []
    i, j = n - 1, m - 1

    while i > 0 or j > 0:
        path.append((i, j))
        if i == 0:
            j -= 1          # só pode vir da esquerda
        elif j == 0:
            i -= 1          # só pode vir de cima
        elif dp[i - 1, j] <= dp[i, j - 1]:
            i -= 1          # veio de cima
        else:
            j -= 1          # veio da esquerda

    path.append((0, 0))
    path.reverse()
    return path


# VARIAÇÃO 1 — TABULAÇÃO BOTTOM-UP

def dp_bottom_up(
    cost: np.ndarray,
    risk: np.ndarray,
    counter: list = None,
) -> dict:
    """
    Programação Dinâmica 2D — Tabulação Bottom-Up.

    Preenche iterativamente a tabela dp[N][M] da origem (0,0) ao destino
    (N-1, M-1), linha por linha, coluna por coluna. Nenhuma recursão —
    cada subproblema é resolvido exatamente uma vez.
    """
    n, m = cost.shape
    if counter is None:
        counter = [0]

    tracemalloc.start()
    t0 = time.perf_counter()

    dp = np.full((n, m), INF, dtype=np.float64)

    # Caso base: origem
    dp[0, 0] = _cell_weight(cost, risk, 0, 0)
    counter[0] += 1

    # Caso base: primeira coluna (só pode vir de cima)
    for i in range(1, n):
        counter[0] += 1
        w = _cell_weight(cost, risk, i, 0)
        if w < INF and dp[i - 1, 0] < INF:
            dp[i, 0] = dp[i - 1, 0] + w

    # Caso base: primeira linha (só pode vir da esquerda)
    for j in range(1, m):
        counter[0] += 1
        w = _cell_weight(cost, risk, 0, j)
        if w < INF and dp[0, j - 1] < INF:
            dp[0, j] = dp[0, j - 1] + w

    # Preenchimento geral: interior da grade
    for i in range(1, n):
        for j in range(1, m):
            counter[0] += 1
            w = _cell_weight(cost, risk, i, j)
            if w < INF:
                melhor_vizinho = min(dp[i - 1, j], dp[i, j - 1])
                if melhor_vizinho < INF:
                    dp[i, j] = melhor_vizinho + w
            # se w == INF (bloqueada) ou vizinhos inacessíveis → dp[i,j] = INF

    t1 = time.perf_counter()
    _, peak  = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    min_cost = dp[n - 1, m - 1]
    optimal_path = reconstruct_path(dp, cost)

    return {
        "dp":           dp,
        "min_cost":     min_cost,
        "optimal_path": optimal_path,
        "feasible":     min_cost < INF,
        "n_iterations": counter[0],
        "time_ms":      (t1 - t0) * 1000,
        "memory_mb":    peak / (1024 ** 2),
        "method":       "bottom_up",
    }


# VARIAÇÃO 2 — MEMOIZAÇÃO TOP-DOWN

def dp_memoization(
    cost: np.ndarray,
    risk: np.ndarray,
    counter: list = None,
) -> dict:
    """
    Programação Dinâmica 2D — Memoização Top-Down.

    Recursão com cache: cada subproblema dp(i,j) é computado apenas na
    primeira vez que é requisitado e armazenado para reuso. O overhead
    de chamadas de função é maior que o bottom-up, mas a lógica espelha
    diretamente a recorrência matemática.

    Usa functools.lru_cache sobre uma função aninhada que captura as
    grades como closures (necessário pois numpy arrays não são hashable).
    """
    n, m = cost.shape
    if counter is None:
        counter = [0]

    tracemalloc.start()
    t0 = time.perf_counter()

    # Aumenta o limite de recursão para grades até 100×100
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, n * m * 2 + 1000))

    @functools.lru_cache(maxsize=None)
    def _dp(i: int, j: int) -> float:
        """Retorna o custo mínimo acumulado de (0,0) até (i,j)."""
        counter[0] += 1
        w = _cell_weight(cost, risk, i, j)
        if w == INF:
            return INF          

        if i == 0 and j == 0:
            return w           

        from_top  = _dp(i - 1, j) if i > 0 else INF
        from_left = _dp(i, j - 1) if j > 0 else INF
        best      = min(from_top, from_left)

        return (best + w) if best < INF else INF

    min_cost = _dp(n - 1, m - 1)

    # Reconstrói a tabela dp completa a partir do cache
    dp = np.full((n, m), INF, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            dp[i, j] = _dp(i, j)

    _dp.cache_clear()
    sys.setrecursionlimit(old_limit)

    t1       = time.perf_counter()
    _, peak  = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    optimal_path = reconstruct_path(dp, cost)

    return {
        "dp":           dp,
        "min_cost":     min_cost,
        "optimal_path": optimal_path,
        "feasible":     min_cost < INF,
        "n_iterations": counter[0],
        "time_ms":      (t1 - t0) * 1000,
        "memory_mb":    peak / (1024 ** 2),
        "method":       "memoization",
    }


# VISUALIZAÇÃO — HEATMAP DA TABELA DP + CAMINHO ÓTIMO

def plot_dp_heatmap(
    dp:           np.ndarray,
    optimal_path: list,
    cost:         np.ndarray,
    blocked:      np.ndarray,
    title:        str = "Tabela DP — Custo Mínimo Acumulado",
    save_path:    str = None,
) -> None:
    """
    Gera o mapa de calor (heatmap) da tabela DP sobre a grade geoespacial,
    com o caminho ótimo destacado em vermelho.

    Elementos visuais:
        - Gradiente azul claro → azul escuro: custo acumulado crescente
        - Cinza escuro: células bloqueadas (cost == -1)
        - Linha vermelha com marcadores: caminho ótimo reconstruído
        - Estrelas ★ na origem e no destino
    """
    import os
    if save_path is None:
        save_path = str(
            Path(__file__).parent.parent / "report" / "dp_heatmap.png"
        )
    os.makedirs(Path(save_path).parent, exist_ok=True)

    n, m = dp.shape

    # Prepara a matriz de visualização: substitui INF pelo máximo finito + 20%
    dp_vis = dp.copy()
    finite_vals = dp_vis[np.isfinite(dp_vis)]
    vmax = finite_vals.max() * 1.1 if len(finite_vals) > 0 else 1.0
    dp_vis[~np.isfinite(dp_vis)] = vmax  # células inacessíveis viram vmax

    # Colormap personalizado: branco → azul
    cmap_base = plt.cm.Blues
    fig, ax   = plt.subplots(figsize=(10, 8))

    im = ax.imshow(dp_vis, cmap=cmap_base, aspect="auto",
                   vmin=0, vmax=vmax, interpolation="nearest")

    # Sobrepõe células bloqueadas em cinza escuro
    bloq_overlay = np.zeros((*dp.shape, 4))
    for i in range(n):
        for j in range(m):
            if blocked[i, j]:
                bloq_overlay[i, j] = [0.25, 0.25, 0.25, 0.85]  # cinza escuro
    ax.imshow(bloq_overlay, aspect="auto", interpolation="nearest")

    # Caminho ótimo: linha vermelha
    if optimal_path:
        path_rows = [p[0] for p in optimal_path]
        path_cols = [p[1] for p in optimal_path]
        ax.plot(path_cols, path_rows,
                color="#EF4444", linewidth=2.5, zorder=5,
                marker="o", markersize=4, markerfacecolor="#EF4444")

        # Marcadores de origem e destino
        ax.plot(path_cols[0],  path_rows[0],
                marker="*", markersize=14, color="#16A34A",
                zorder=6, label="Origem (0,0)")
        ax.plot(path_cols[-1], path_rows[-1],
                marker="*", markersize=14, color="#DC2626",
                zorder=6, label="Destino (N-1,M-1)")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Custo acumulado mínimo dp[i][j]", fontsize=10)

    # Legenda
    patch_bloq   = mpatches.Patch(color="#404040", label="Célula bloqueada")
    patch_caminho = mpatches.Patch(color="#EF4444", label="Caminho ótimo")
    handles = [patch_bloq, patch_caminho]
    if optimal_path:
        handles += ax.get_legend_handles_labels()[0]
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              framealpha=0.85)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Coluna j  →  Leste", fontsize=10)
    ax.set_ylabel("Linha i  →  Sul", fontsize=10)

    step = max(1, n // 10)
    ax.set_xticks(range(0, m, step))
    ax.set_yticks(range(0, n, step))

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Heatmap salvo em: {save_path}")


# DEMONSTRAÇÃO E COMPARAÇÃO

def run_demo() -> None:
    """
    Executa e compara os dois métodos DP nos dois cenários,
    valida resultados contra si mesmos e gera os heatmaps.
    """
    print("\n" + "=" * 60)
    print("  Programação Dinâmica 2D — Demonstração e Comparação")
    print("=" * 60)

    # --- Validação cruzada: grade pequena (5×5) ---
    print("\n  [1/3] Validação cruzada bottom-up vs memoização (5x5)...")
    for cenario in ["A", "C"]:
        grid = generate_small_grid(n=5, m=5, cenario=cenario, seed=7)
        r_bu  = dp_bottom_up(grid["cost"], grid["risk"])
        r_mem = dp_memoization(grid["cost"], grid["risk"])

        match = abs(r_bu["min_cost"] - r_mem["min_cost"]) < 1e-9
        status = "OK" if match else "DIVERGÊNCIA"
        print(f"     Cenário {cenario}: bottom-up={r_bu['min_cost']:.4f} | "
              f"memo={r_mem['min_cost']:.4f} | {status}")

    # --- Cenários reais (50×50) ---
    print("\n  [2/3] Rodando nos cenários reais (50x50)...")

    for letra in ["A", "C"]:
        try:
            grid = load_grid(letra)
        except FileNotFoundError:
            print(f"     Grade {letra} não encontrada. "
                  "Execute: python src/data_loader.py")
            continue

        cost    = grid["cost"]
        risk    = grid["risk"]
        blocked = grid["blocked"]
        meta    = grid["metadata"]

        print(f"\n  {'─'*55}")
        print(f"  Cenário {letra} — {meta['nome']}")
        print(f"  {'─'*55}")

        for method_fn, label in [
            (dp_bottom_up,   "Bottom-Up  "),
            (dp_memoization, "Memoização "),
        ]:
            res = method_fn(cost, risk)
            status = "viável" if res["feasible"] else "sem caminho"
            print(f"     {label}: custo={res['min_cost']:9.3f} | "
                  f"iter={res['n_iterations']:6d} | "
                  f"t={res['time_ms']:7.2f}ms | "
                  f"mem={res['memory_mb']:.4f}MB | {status}")

        # Heatmap do bottom-up (mais limpo visualmente)
        res_bu = dp_bottom_up(cost, risk)
        print(f"\n  [3/3] Gerando heatmap — Cenário {letra}...")
        plot_dp_heatmap(
            dp           = res_bu["dp"],
            optimal_path = res_bu["optimal_path"],
            cost         = cost,
            blocked      = blocked,
            title        = (
                f"DP — Custo Mínimo Acumulado | Cenário {letra}\n"
                f"{meta['nome']}"
            ),
            save_path = str(
                Path(__file__).parent.parent / "report"
                / f"dp_heatmap_cenario_{letra}.png"
            ),
        )

        if res_bu["feasible"]:
            print(f"     Custo ótimo:   {res_bu['min_cost']:.4f}")
            print(f"     Tamanho path:  {len(res_bu['optimal_path'])} células")
            print(f"     Início path:   {res_bu['optimal_path'][:3]}...")
            print(f"     Fim path:      ...{res_bu['optimal_path'][-3:]}")

    print("\n  Programação Dinâmica concluída.")


# EXECUÇÃO DIRETA

if __name__ == "__main__":
    run_demo()