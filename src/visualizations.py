"""
visualizations.py — Geração Centralizada de Figuras Obrigatórias


Papel no sistema:
    Ponto único de entrada para gerar todas as figuras exigidas pelo
    enunciado (Seção 6.1). Importa as funções de plotagem dos módulos
    de cada algoritmo e adiciona a figura de análise de sensibilidade.

Figuras obrigatórias:
    [1] Heatmap da tabela DP + caminho ótimo destacado
        → src/dynamic_programming.py :: plot_dp_heatmap()
        → report/dp_heatmap_cenario_A.png
        → report/dp_heatmap_cenario_C.png

    [2] Histograma + boxplot da distribuição Monte Carlo
        → src/monte_carlo.py :: plot_mc_distribution()
        → report/mc_distribuicao_cenario_A.png
        → report/mc_distribuicao_cenario_C.png

    [3] Curva de escalabilidade: tempo x N (3 algoritmos)
        → src/performance_monitor.py :: plot_escalabilidade()
        → report/escalabilidade.png

    [4] Scatter: custo ótimo x tempo computacional
        → src/performance_monitor.py :: plot_scatter_custo_tempo()
        → report/scatter_custo_tempo.png

    [5] Análise de sensibilidade: variação da solução p±20%
        → src/visualizations.py :: plot_sensitivity_comparison()   [NOVO]
        → report/sensibilidade_comparativa.png
"""

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent))
from data_loader         import load_grid
from dynamic_programming import dp_bottom_up, plot_dp_heatmap
from monte_carlo         import run_monte_carlo, sensitivity_analysis, plot_mc_distribution
from performance_monitor import run_benchmark, plot_escalabilidade, plot_scatter_custo_tempo

REPORT_DIR = Path(__file__).parent.parent / "report"
K_MC = 10_000

# FIGURA 5 — ANÁLISE DE SENSIBILIDADE COMPARATIVA (NOVA)

def plot_sensitivity_comparison(
    sens_A:    dict,
    sens_C:    dict,
    save_path: str = None,
) -> None:
    """
    Gera a Figura 5 obrigatória: análise de sensibilidade p[i][j] ± 20%.

    Layout (2 x 2):
        [0,0] Violins Cenário A   [0,1] Violins Cenário C
        [1,0] Barras erro Cen. A  [1,1] Barras erro Cen. C
    """
    if save_path is None:
        save_path = str(REPORT_DIR / "sensibilidade_comparativa.png")
    os.makedirs(Path(save_path).parent, exist_ok=True)

    delta   = sens_A["delta"]
    configs = ["otimista", "base", "pessimista"]
    labels  = [
        f"Otimista\n(p×{1-delta:.0%})",
        "Base\n(p histórico)",
        f"Pessimista\n(p×{1+delta:.0%})",
    ]
    cores = ["#16A34A", "#2563EB", "#DC2626"]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    for col, (sens, letra, nome_cenario) in enumerate([
        (sens_A, "A", "Seca no Cerrado/Nordeste — MATOPIBA"),
        (sens_C, "C", "Desmatamento na Amazônia — DETER/PRODES"),
    ]):
        # Painel superior: violin plot
        ax_v = fig.add_subplot(gs[0, col])

        dados_violin = [sens[cfg]["costs_valid"] for cfg in configs]

        parts = ax_v.violinplot(
            dados_violin,
            positions=range(len(configs)),
            showmeans=True,
            showmedians=True,
            showextrema=True,
        )

        # Colorir cada violino
        for i, (pc, cor) in enumerate(zip(parts["bodies"], cores)):
            pc.set_facecolor(cor)
            pc.set_alpha(0.55)
            pc.set_edgecolor(cor)

        parts["cmeans"].set_color("black")
        parts["cmeans"].set_linewidth(2)
        parts["cmedians"].set_color("white")
        parts["cmedians"].set_linewidth(1.5)

        # Anotações de média sobre cada violino
        for i, cfg in enumerate(configs):
            m = sens[cfg]["mean"]
            ax_v.annotate(
                f"μ={m:.1f}",
                xy=(i, m),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center", fontsize=8,
                color=cores[i], fontweight="bold",
            )

        ax_v.set_xticks(range(len(configs)))
        ax_v.set_xticklabels(labels, fontsize=9)
        ax_v.set_ylabel("Custo ótimo simulado", fontsize=9)
        ax_v.set_title(
            f"Cenário {letra} — Distribuição dos Custos\n{nome_cenario}",
            fontsize=10, fontweight="bold",
        )
        ax_v.grid(True, axis="y", alpha=0.3)

        # Painel inferior: barras de erro + impacto %
        ax_b = fig.add_subplot(gs[1, col])

        medias  = [sens[cfg]["mean"]     for cfg in configs]
        ic_low  = [sens[cfg]["ic95_low"] for cfg in configs]
        ic_high = [sens[cfg]["ic95_high"]for cfg in configs]
        erros_baixo = [m - l for m, l in zip(medias, ic_low)]
        erros_cima  = [h - m for m, h in zip(medias, ic_high)]

        xs = range(len(configs))
        barras = ax_b.bar(xs, medias, color=cores, alpha=0.72,
                          edgecolor="white", linewidth=0.8, width=0.5)
        ax_b.errorbar(
            xs, medias,
            yerr=[erros_baixo, erros_cima],
            fmt="none", color="black", capsize=6, linewidth=1.8, zorder=5,
        )

        # Linha de referência: valor base
        base_mean = sens["base"]["mean"]
        ax_b.axhline(base_mean, color="#2563EB", lw=1.3, ls="--", alpha=0.7,
                     label=f"Base = {base_mean:.2f}")

        # Anotações de impacto percentual
        for i, (cfg, m) in enumerate(zip(configs, medias)):
            delta_pct = (m - base_mean) / base_mean * 100
            sinal     = "+" if delta_pct >= 0 else ""
            cor_txt   = "#DC2626" if delta_pct > 0 else ("#16A34A" if delta_pct < 0 else "#6B7280")
            ax_b.annotate(
                f"{sinal}{delta_pct:.2f}%",
                xy=(i, m + max(erros_cima[i], 1)),
                xytext=(0, 6), textcoords="offset points",
                ha="center", fontsize=9,
                color=cor_txt, fontweight="bold",
            )

        # Desvios padrão na legenda interna
        stats_txt = "\n".join(
            f"{lbl.replace(chr(10),' ')}: σ={sens[cfg]['std']:.2f}"
            for lbl, cfg in zip(["Otimista", "Base", "Pessimista"], configs)
        )
        ax_b.text(
            0.98, 0.05, stats_txt, transform=ax_b.transAxes,
            fontsize=7.5, va="bottom", ha="right", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
        )

        ax_b.set_xticks(xs)
        ax_b.set_xticklabels(labels, fontsize=9)
        ax_b.set_ylabel("Custo médio ± IC 95%", fontsize=9)
        ax_b.set_title(
            f"Cenário {letra} — Impacto Percentual vs. Base",
            fontsize=10, fontweight="bold",
        )
        ax_b.legend(fontsize=8, loc="upper left")
        ax_b.grid(True, axis="y", alpha=0.3)

        # Ajusta ylim para dar espaço às anotações
        ymax = max(ic_high) * 1.08
        ymin = min(ic_low)  * 0.94
        ax_b.set_ylim(ymin, ymax)

    fig.suptitle(
        f"Análise de Sensibilidade — Impacto de p[i][j] ± {delta*100:.0f}%\n"
        f"Cenários A (MATOPIBA) e C (Amazônia) | K={sens_A['base']['K']:,} cenários MC",
        fontsize=13, fontweight="bold", y=1.01,
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Sensibilidade comparativa salva em: {save_path}")


# FUNÇÃO PRINCIPAL — GERA TODAS AS FIGURAS

def generate_all(k_mc: int = K_MC) -> None:
    """
    Gera todas as 5 figuras obrigatórias do enunciado em sequência.

    Ordem de execução:
        1. Carrega grades dos dois cenários
        2. Figuras 1: Heatmaps DP (A e C)
        3. Figuras 2: Histogramas MC (A e C)
        4. Figura  5: Sensibilidade comparativa (A e C)
        5. Benchmark de desempenho (necessário para Figs. 3 e 4)
        6. Figura  3: Curva de escalabilidade
        7. Figura  4: Scatter custo x tempo
    """
    os.makedirs(REPORT_DIR, exist_ok=True)

    print("\n" + "=" * 62)
    print("  Visualizações — Gerando todas as figuras obrigatórias")
    print("=" * 62)

    # Carrega grades
    grids = {}
    for letra in ["A", "C"]:
        try:
            grids[letra] = load_grid(letra)
            print(f"\n  Grade {letra} carregada: {grids[letra]['metadata']['nome']}")
        except FileNotFoundError:
            print(f"\n  Grade {letra} não encontrada.")
            print("       Execute: python src/data_loader.py")
            return

    # Figura 1: Heatmaps DP
    print("\n  [Figura 1/5] Heatmaps DP + caminho ótimo...")
    for letra in ["A", "C"]:
        g   = grids[letra]
        res = dp_bottom_up(g["cost"], g["risk"])
        plot_dp_heatmap(
            dp           = res["dp"],
            optimal_path = res["optimal_path"],
            cost         = g["cost"],
            blocked      = g["blocked"],
            title        = (
                f"DP — Custo Mínimo Acumulado | Cenário {letra}\n"
                f"{g['metadata']['nome']}"
            ),
            save_path    = str(REPORT_DIR / f"dp_heatmap_cenario_{letra}.png"),
        )

    # Figura 2: Histogramas + Boxplots MC 
    print("\n  [Figura 2/5] Histogramas + Boxplots Monte Carlo...")
    for letra in ["A", "C"]:
        g = grids[letra]
        print(f"\n     Cenário {letra} — simulação base (K={k_mc:,})...")
        result_base = run_monte_carlo(
            cost=g["cost"], risk=g["risk"], prob=g["prob"],
            K=k_mc, seed=42, verbose=True,
        )
        print(f"\n     Cenário {letra} — análise de sensibilidade (±20%)...")
        result_sens = sensitivity_analysis(
            cost=g["cost"], risk=g["risk"], prob=g["prob"],
            K=k_mc, delta=0.20, seed=99,
        )
        plot_mc_distribution(
            result_base  = result_base,
            result_sens  = result_sens,
            cenario_nome = g["metadata"]["nome"],
            save_path    = str(REPORT_DIR / f"mc_distribuicao_cenario_{letra}.png"),
        )
        # Guarda sens para Figura 5
        grids[letra]["sens"] = result_sens

    # Figura 5: Sensibilidade Comparativa
    print("\n  [Figura 5/5] Análise de sensibilidade comparativa...")
    plot_sensitivity_comparison(
        sens_A    = grids["A"]["sens"],
        sens_C    = grids["C"]["sens"],
        save_path = str(REPORT_DIR / "sensibilidade_comparativa.png"),
    )

    # Figuras 3 e 4: Escalabilidade + Scatter
    print("\n  [Figuras 3 e 4/5] Benchmark de desempenho...")
    resultados = run_benchmark()
    plot_escalabilidade(
        resultados,
        save_path = str(REPORT_DIR / "escalabilidade.png"),
    )
    plot_scatter_custo_tempo(
        resultados,
        save_path = str(REPORT_DIR / "scatter_custo_tempo.png"),
    )

    # Resumo final
    figuras = [
        "dp_heatmap_cenario_A.png",
        "dp_heatmap_cenario_C.png",
        "mc_distribuicao_cenario_A.png",
        "mc_distribuicao_cenario_C.png",
        "sensibilidade_comparativa.png",
        "escalabilidade.png",
        "scatter_custo_tempo.png",
    ]

    print("\n" + "=" * 62)
    print("  Todas as figuras geradas com sucesso!")
    print("=" * 62)
    for fig_nome in figuras:
        caminho = REPORT_DIR / fig_nome
        existe  = "✅" if caminho.exists() else "❌"
        tamanho = f"{caminho.stat().st_size / 1024:.0f} KB" if caminho.exists() else "—"
        print(f"     {existe}  report/{fig_nome:<45} {tamanho:>8}")

    print("\n     Próximo passo: pytest tests/\n")


# EXECUÇÃO DIRETA

if __name__ == "__main__":
    generate_all()