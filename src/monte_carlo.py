"""
monte_carlo.py — Simulação de Incerteza Climática

Papel no sistema:
    Quantifica a incerteza climática inerente às probabilidades p[i][j],
    estimadas a partir de séries temporais satelitais 2000-2024.
    Para cada cenário simulado, as probabilidades são perturbadas e a DP
    é reexecutada, gerando a distribuição empírica do custo ótimo.

Performance:
    A DP vetorizada processa K cenários em lotes (batches) usando
    numpy, evitando loop Python explícito. Para K=10.000, N=M=50:
    tempo estimado < 60s numa máquina moderna.
"""

import sys
import time
import tracemalloc
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from data_loader    import load_grid
from dynamic_programming import dp_bottom_up

INF = float("inf")

# DP VETORIZADA — NÚCLEO DO MONTE CARLO

def _dp_vectorized_batch(
    cost:        np.ndarray,
    risk:        np.ndarray,
    prob_batch:  np.ndarray,
) -> np.ndarray:
    """
    Executa a DP bottom-up em lote para um batch de cenários simultaneamente.

    Ao invés de chamar dp_bottom_up() K vezes,
    esta função processa B cenários em paralelo usando arrays 3D do numpy.

    Fórmula do peso efetivo por cenário k:
        w_k[i][j] = cost[i][j] x (1 + risk[i][j] x p'_k[i][j])
    """
    n, m = cost.shape
    B = prob_batch.shape[0]

    # Máscara de células bloqueadas (expandida para o batch)
    bloq_mask = (cost == -1)                        
    bloq_3d   = bloq_mask[np.newaxis, :, :]       

    # Peso efetivo: (B, N, M)
    # w_k[i,j] = cost[i,j] × (1 + risk[i,j] × p'_k[i,j])
    cost_f = cost.astype(np.float64)                # (N, M)
    w = cost_f[np.newaxis] * (
        1.0 + risk[np.newaxis] * prob_batch
    )                                              
    w[bloq_3d.repeat(B, axis=0)] = INF           

    # Tabela DP: (B, N, M)
    dp = np.full((B, n, m), INF, dtype=np.float64)
    dp[:, 0, 0] = w[:, 0, 0]

    # Primeira coluna: acumula para baixo
    for i in range(1, n):
        mask = (dp[:, i - 1, 0] < INF) & (w[:, i, 0] < INF)
        dp[:, i, 0] = np.where(mask, dp[:, i - 1, 0] + w[:, i, 0], INF)

    # Primeira linha: acumula para a direita
    for j in range(1, m):
        mask = (dp[:, 0, j - 1] < INF) & (w[:, 0, j] < INF)
        dp[:, 0, j] = np.where(mask, dp[:, 0, j - 1] + w[:, 0, j], INF)

    # Interior: recorrência principal
    for i in range(1, n):
        for j in range(1, m):
            melhor = np.minimum(dp[:, i - 1, j], dp[:, i, j - 1])  # (B,)
            viavel = (melhor < INF) & (w[:, i, j] < INF)
            dp[:, i, j] = np.where(viavel, melhor + w[:, i, j], INF)

    return dp[:, n - 1, m - 1]


# SIMULAÇÃO MONTE CARLO PRINCIPAL

def run_monte_carlo(
    cost:          np.ndarray,
    risk:          np.ndarray,
    prob:          np.ndarray,
    K:             int   = 10_000,
    concentration: float = 8.0,
    batch_size:    int   = 500,
    seed:          int   = 42,
    verbose:       bool  = True,
) -> dict:
    """
    Simula a incerteza climática via Monte Carlo (K cenários).

    Para cada cenário k:
        p'_k[i][j] ~ Beta(a=p[i][j]xc, β=(1-p[i][j])xc)
        onde c = concentration (controla a variância da amostragem)

    Processa em batches para eficiência de memória.
    """
    rng = np.random.default_rng(seed)
    n, m = cost.shape

    # Pré-calcula a e b para toda a grade de uma vez
    p_clipped = np.clip(prob, 0.05, 0.95)           
    alpha = p_clipped * concentration              
    beta  = (1.0 - p_clipped) * concentration     

    all_costs  = np.empty(K, dtype=np.float64)
    n_batches  = (K + batch_size - 1) // batch_size

    t0 = time.perf_counter()

    for b in range(n_batches):
        k_start = b * batch_size
        k_end   = min(k_start + batch_size, K)
        B       = k_end - k_start

        # Amostra B cenários de probabilidade: Beta(a, b) por célula
        u = rng.random(size=(B, n, m))             
        prob_batch = rng.beta(
            alpha[np.newaxis].repeat(B, axis=0), # (B, N, M)
            beta[np.newaxis].repeat(B, axis=0),
        )

        batch_costs = _dp_vectorized_batch(cost, risk, prob_batch)
        all_costs[k_start:k_end] = batch_costs

        if verbose and (b + 1) % max(1, n_batches // 5) == 0:
            pct = (b + 1) / n_batches * 100
            elapsed = time.perf_counter() - t0
            print(f"     Lote {b+1:3d}/{n_batches} ({pct:5.1f}%) — "
                  f"{elapsed:.1f}s decorridos")

    t1 = time.perf_counter()

    n_infeasible = int(np.sum(all_costs == INF))
    costs_valid  = all_costs[all_costs < INF]

    if len(costs_valid) == 0:
        raise RuntimeError("Nenhum cenário Monte Carlo gerou caminho viável.")

    # Estatísticas
    mean_c   = float(np.mean(costs_valid))
    median_c = float(np.median(costs_valid))
    std_c    = float(np.std(costs_valid))

    ic_low  = float(np.percentile(costs_valid, 2.5))
    ic_high = float(np.percentile(costs_valid, 97.5))

    if verbose:
        print(f"\n     Monte Carlo concluído em {t1 - t0:.1f}s")
        print(f"     Cenários totais:    {K:,}")
        print(f"     Inviáveis (INF):    {n_infeasible:,} "
              f"({n_infeasible/K*100:.1f}%)")
        print(f"     Média:              {mean_c:.4f}")
        print(f"     Mediana:            {median_c:.4f}")
        print(f"     Desvio padrão:      {std_c:.4f}")
        print(f"     IC 95%:             [{ic_low:.4f}, {ic_high:.4f}]")

    return {
        "costs":        all_costs,
        "costs_valid":  costs_valid,
        "mean":         mean_c,
        "median":       median_c,
        "std":          std_c,
        "ic95_low":     ic_low,
        "ic95_high":    ic_high,
        "n_infeasible": n_infeasible,
        "time_s":       t1 - t0,
        "K":            K,
    }


# ANÁLISE DE SENSIBILIDADE

def sensitivity_analysis(
    cost:          np.ndarray,
    risk:          np.ndarray,
    prob:          np.ndarray,
    K:             int   = 10_000,
    delta:         float = 0.20,
    seed:          int   = 99,
) -> dict:
    """
    Analisa o impacto de variações em p[i][j] sobre a distribuição de custos.

    Executa 3 simulações Monte Carlo:
        - Pessimista: p x (1 + delta), capped em 0.95
        - Base:       p original
        - Otimista:   p x (1 - delta)
    """
    configs = {
        "pessimista": np.clip(prob * (1 + delta), 0.05, 0.95),
        "base":       prob,
        "otimista":   np.clip(prob * (1 - delta), 0.05, 0.95),
    }

    resultados = {}
    for nome, p_var in configs.items():
        print(f"\n   Sensibilidade — {nome} (p × {1 + delta if nome == 'pessimista' else (1 - delta if nome == 'otimista' else 1.0):.2f}):")
        resultados[nome] = run_monte_carlo(
            cost=cost, risk=risk, prob=p_var,
            K=K, seed=seed, verbose=True,
        )

    resultados["delta"] = delta
    return resultados


# VISUALIZAÇÃO — HISTOGRAMA + BOXPLOT

def plot_mc_distribution(
    result_base:      dict,
    result_sens:      dict,
    cenario_nome:     str  = "Cenário",
    save_path:        str  = None,
) -> None:
    """
    Gera a figura obrigatória: histograma + boxplot da distribuição MC,
    com análise de sensibilidade sobreposta.

    Layout:
        Linha superior: histogramas das 3 simulações (base + ±20%)
        Linha inferior: boxplot comparativo das 3 distribuições
    """
    if save_path is None:
        save_path = str(
            Path(__file__).parent.parent / "report" / "mc_distribuicao.png"
        )
    os.makedirs(Path(save_path).parent, exist_ok=True)

    cores = {
        "pessimista": "#DC2626",  
        "base":       "#2563EB",  
        "otimista":   "#16A34A",  
    }
    labels = {
        "pessimista": f"Pessimista (p x {1 + result_sens['delta']:.0%})",
        "base":       "Base (p histórico)",
        "otimista":   f"Otimista (p x {1 - result_sens['delta']:.0%})",
    }

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # histogramas (linha superior)
    for col, (nome, cor) in enumerate(cores.items()):
        ax = fig.add_subplot(gs[0, col])
        res = result_sens[nome]
        cv  = res["costs_valid"]

        ax.hist(cv, bins=60, color=cor, alpha=0.75, edgecolor="white", lw=0.4)

        # Linha da média
        ax.axvline(res["mean"],     color="black",  lw=1.8, ls="-",  label=f"Média={res['mean']:.1f}")
        ax.axvline(res["median"],   color="gray",   lw=1.4, ls="--", label=f"Mediana={res['median']:.1f}")
        ax.axvline(res["ic95_low"], color=cor,      lw=1.2, ls=":",  alpha=0.9)
        ax.axvline(res["ic95_high"],color=cor,      lw=1.2, ls=":",  alpha=0.9,
                   label=f"IC95% [{res['ic95_low']:.1f}, {res['ic95_high']:.1f}]")

        # Faixa IC
        ax.axvspan(res["ic95_low"], res["ic95_high"], alpha=0.10, color=cor)

        ax.set_title(labels[nome], fontsize=10, fontweight="bold", color=cor)
        ax.set_xlabel("Custo ótimo simulado", fontsize=8)
        ax.set_ylabel("Frequência", fontsize=8)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)

        # Caixa de estatísticas
        stats_txt = (f"μ={res['mean']:.2f}\n"
                     f"σ={res['std']:.2f}\n"
                     f"n={res['K']:,}\n"
                     f"inviáveis={res['n_infeasible']}")
        ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
                fontsize=7, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    # boxplot comparativo
    ax_box = fig.add_subplot(gs[1, :])

    dados_box  = [result_sens[n]["costs_valid"] for n in ["pessimista", "base", "otimista"]]
    labels_box = [labels[n] for n in ["pessimista", "base", "otimista"]]
    cores_box  = [cores[n]  for n in ["pessimista", "base", "otimista"]]

    bp = ax_box.boxplot(
        dados_box,
        vert=False,
        patch_artist=True,
        notch=True,
        widths=0.55,
        medianprops=dict(color="white", linewidth=2.5),
        whiskerprops=dict(linewidth=1.4),
        capprops=dict(linewidth=1.4),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )

    for patch, cor in zip(bp["boxes"], cores_box):
        patch.set_facecolor(cor)
        patch.set_alpha(0.75)

    # Linha de referência
    custo_det = result_base["mean"]
    ax_box.axvline(custo_det, color="black", lw=1.5, ls="--",
                   label=f"Custo determinístico DP = {custo_det:.1f}", zorder=5)

    ax_box.set_yticks(range(1, 4))
    ax_box.set_yticklabels(labels_box, fontsize=9)
    ax_box.set_xlabel("Custo ótimo simulado", fontsize=10)
    ax_box.set_title(
        "Comparativo de Sensibilidade — Boxplot das Distribuições MC",
        fontsize=11, fontweight="bold"
    )
    ax_box.legend(fontsize=9, loc="lower right")
    ax_box.grid(True, axis="x", alpha=0.3)

    fig.suptitle(
        f"Monte Carlo — Distribuição do Custo Ótimo sob Incerteza Climática\n"
        f"{cenario_nome} | K={result_base['K']:,} cenários | "
        f"Perturbação: +/-{result_sens['delta']*100:.0f}%",
        fontsize=12, fontweight="bold", y=1.01,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Figura MC salva em: {save_path}")


# DEMONSTRAÇÃO NOS DOIS CENÁRIOS

def run_demo() -> None:
    """
    Executa a simulação Monte Carlo completa nos dois cenários,
    incluindo análise de sensibilidade e geração das figuras obrigatórias.
    """
    print("\n" + "=" * 60)
    print("  Monte Carlo — Simulação de Incerteza Climática")
    print("=" * 60)

    K = 10_000

    for letra in ["A", "C"]:
        try:
            grid = load_grid(letra)
        except FileNotFoundError:
            print(f"\n  Grade {letra} não encontrada. "
                  "Execute: python src/data_loader.py")
            continue

        cost    = grid["cost"]
        risk    = grid["risk"]
        prob    = grid["prob"]
        meta    = grid["metadata"]

        print(f"\n{'=' * 60}")
        print(f"  Cenário {letra} — {meta['nome']}")
        print(f"{'=' * 60}")

        # Simulação base
        print(f"\n  [1/3] Simulação base (K={K:,})...")
        result_base = run_monte_carlo(
            cost=cost, risk=risk, prob=prob,
            K=K, seed=42, verbose=True,
        )

        # Análise de sensibilidade
        print(f"\n  [2/3] Análise de sensibilidade (±20%)...")
        result_sens = sensitivity_analysis(
            cost=cost, risk=risk, prob=prob,
            K=K, delta=0.20, seed=99,
        )

        # Figura obrigatória
        print(f"\n  [3/3] Gerando figura histograma + boxplot...")
        plot_mc_distribution(
            result_base  = result_base,
            result_sens  = result_sens,
            cenario_nome = meta["nome"],
            save_path    = str(
                Path(__file__).parent.parent / "report"
                / f"mc_distribuicao_cenario_{letra}.png"
            ),
        )

        # Resumo
        print(f"\n  ── Resumo Cenário {letra} ──────────────────────────────")
        print(f"  Configuração     │ Média     │ DP       │ IC 95%")
        print(f"  {'─'*52}")
        for nome in ["otimista", "base", "pessimista"]:
            r = result_sens[nome]
            print(f"  {nome:<16} │ {r['mean']:9.3f} │ {r['std']:8.3f} │ "
                  f"[{r['ic95_low']:.3f}, {r['ic95_high']:.3f}]")

        delta_pess_base = (
            (result_sens["pessimista"]["mean"] - result_sens["base"]["mean"])
            / result_sens["base"]["mean"] * 100
        )
        delta_otim_base = (
            (result_sens["base"]["mean"] - result_sens["otimista"]["mean"])
            / result_sens["base"]["mean"] * 100
        )
        print(f"\n  Impacto +20% prob: +{delta_pess_base:.2f}% no custo médio")
        print(f"  Impacto -20% prob: -{delta_otim_base:.2f}% no custo médio")

    print("\n  Monte Carlo concluído.")
    print("    Próximo passo: python src/performance_monitor.py\n")


# EXECUÇÃO DIRETA

if __name__ == "__main__":
    run_demo()