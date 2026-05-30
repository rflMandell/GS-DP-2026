"""
Coleta e Processamento de Dados Reais

Estrategia de dados:
Valores reais extraidos de fontes publicas documentadas sao usados como parametros para gerar grades NxM com distribuicoes estatisticas calibradas
"""

import os
import json
import numpy as np
from pathlib import Path

# parametros globais
GRID_SIZE = 50

# baseline
#cenario A: 
MATOPIBA_BASELINES = {
    # ndvi - precip_mm - prob_seca - idh
    "maranhao":  (0.48,   1350,      0.28,    0.639),
    "tocantins": (0.42,   1200,      0.35,    0.699),
    "piaui":     (0.35,    850,      0.52,    0.646),
    "bahia":     (0.38,    950,      0.45,    0.660),
}

#cenario C:
AMAZONIA_BASELINES = {
    # frac_desmat - prob_alerta - frac_fluvial
    "para":          (0.40,         0.68,        0.35),
    "amazonas":      (0.20,         0.45,        0.58),
    "mato_grosso":   (0.15,         0.52,        0.12),
    "rondonia":      (0.12,         0.61,        0.28),
    "outros":        (0.13,         0.30,        0.20),
}

#fracao esperada de celulas blockeadas no cenario C
AMAZONIA_BLOCKED_FRACTION = sum(
    fd * ff for fd, _, ff in AMAZONIA_BASELINES.values()
)


#Funcoes auxiliares

def _spatial_noise(ni: int, nj: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """
    Gera ruido espacial suave
    simula a variacao continua de condicoes climaticas no espaco geografico
    """
    
    xx, yy = np.meshgrid(np.linspace(0, np.pi, nj), np.linspace(0, np.pi, ni))
    senoidal = np.sin(xx) * np.cos(yy)
    guassiano = rng.normal(0, 0.3, size=(ni, nj))
    noise = amplitude * (0.6 * senoidal + 0.4 * guassiano)
    return noise


def _beta_params(mean: float, concentration: float = 8.0) -> tuple[float, float]:
    """
    Calcula a e b da distribuição Beta a partir da média e concentração.
    Concentração alta → distribuição mais estreita em torno da média.

    Fórmula: a = mean x concentration, β = (1 - mean) x concentration
    """
    mean = np.clip(mean, 0.05, 0.95)  # evita α ou β = 0
    return mean * concentration, (1 - mean) * concentration


# CENÁRIO A — MATOPIBA (SECA NO CERRADO/NORDESTE)

def generate_grid_matopiba(n: int = GRID_SIZE, m: int = GRID_SIZE, seed: int = 42) -> dict:
    """
    Gera a grade NxM do Cenário A — Seca no Cerrado/Nordeste (MATOPIBA).

    Estrutura da grade:
        Dividida em 4 quadrantes correspondendo às sub-regiões MA, TO, PI, BA.
        Cada célula representa um município ou célula de ~10 km² da região.

    Returns:
        dict com 'risk', 'cost', 'prob', 'blocked', 'metadata'
    """
    rng = np.random.default_rng(seed)

    risk    = np.zeros((n, m), dtype=np.float64)
    cost    = np.zeros((n, m), dtype=np.float64)
    prob    = np.zeros((n, m), dtype=np.float64)
    blocked = np.zeros((n, m), dtype=bool)

    # Quadrantes: [MA noroeste | TO nordeste | BA sudoeste | PI sudeste]
    regioes   = list(MATOPIBA_BASELINES.values())
    quadrantes = [
        (slice(0,     n // 2), slice(0,     m // 2)),  # MA
        (slice(0,     n // 2), slice(m // 2, m    )),  # TO
        (slice(n // 2, n    ), slice(0,     m // 2)),  # BA
        (slice(n // 2, n    ), slice(m // 2, m    )),  # PI
    ]

    for (ndvi, precip, prob_seca, idh), (qi, qj) in zip(regioes, quadrantes):
        ni = qi.stop - qi.start
        nj = qj.stop - qj.start

        r_mean = 1.0 - ndvi  # NDVI alto → risco baixo
        a, b   = _beta_params(r_mean, concentration=10.0)
        base_risk = rng.beta(a, b, size=(ni, nj))
        noise     = _spatial_noise(ni, nj, amplitude=0.07, rng=rng)
        risk[qi, qj] = np.clip(base_risk + noise, 0.0, 1.0)

        custo_base  = 1.0 + 9.0 * (1.0 - idh)
        custo_noise = rng.normal(0, 0.6, size=(ni, nj))
        cost[qi, qj] = np.clip(custo_base + custo_noise, 1.0, 10.0)

        a_p, b_p       = _beta_params(prob_seca, concentration=8.0)
        prob[qi, qj]   = rng.beta(a_p, b_p, size=(ni, nj))

    cost_int = np.round(cost).astype(int)

    return {
        "risk":    risk,
        "cost":    cost_int,
        "prob":    prob,
        "blocked": blocked,
        "metadata": {
            "cenario":          "A",
            "nome":             "Seca no Cerrado/Nordeste — MATOPIBA",
            "tamanho":          f"{n}x{m}",
            "blocked_fraction": 0.0,
            "seed":             seed,
            "fontes": [
                "MODIS MOD13A3 — NDVI médio por sub-região 2000-2024",
                "INMET BDMEP — Normais climatológicas 1991-2020",
                "IBGE/PNUD — Atlas Desenvolvimento Humano Municipal 2021",
            ],
        },
    }


# CENÁRIO C — AMAZÔNIA (DESMATAMENTO DETER/PRODES)

def generate_grid_amazonia(n: int = GRID_SIZE, m: int = GRID_SIZE, seed: int = 42) -> dict:
    """
    Gera a grade NxM do Cenário C — Desmatamento na Amazônia (DETER/PRODES).

    Estrutura da grade:
        Faixas horizontais proporcional à contribuição de cada estado no
        desmatamento total PRODES 2023. Célula representa ~10 km² do
        arco do desmatamento (PA → AM → MT → RO → demais).

    Returns:
        dict com 'risk', 'cost', 'prob', 'blocked', 'metadata'
    """
    rng = np.random.default_rng(seed)

    risk    = np.zeros((n, m), dtype=np.float64)
    cost    = np.zeros((n, m), dtype=np.float64)
    prob    = np.zeros((n, m), dtype=np.float64)
    blocked = np.zeros((n, m), dtype=bool)

    estados = list(AMAZONIA_BASELINES.items())
    fracoes = np.array([v[0] for _, v in estados])
    limites = np.round(np.cumsum(fracoes) * n).astype(int)
    limites = np.concatenate([[0], np.minimum(limites, n)])
    limites[-1] = n  # última faixa sempre fecha em n

    # Gradiente leste-oeste: borda leste tem maior pressão de desmatamento
    gradiente_col = np.linspace(0.85, 1.15, m)

    for idx, (estado, (frac_desmat, prob_alerta, frac_fluvial)) in enumerate(estados):
        i0, i1 = int(limites[idx]), int(limites[idx + 1])
        if i0 >= i1:
            continue
        ni = i1 - i0

        # --- Risco: proporcional à taxa de desmatamento ---
        r_mean    = min(frac_desmat * 2.0, 0.90)
        a_r, b_r  = _beta_params(r_mean, concentration=7.0)
        base_risk = rng.beta(a_r, b_r, size=(ni, m))
        noise     = _spatial_noise(ni, m, amplitude=0.06, rng=rng)
        risco_faixa = np.clip(base_risk * gradiente_col + noise, 0.0, 1.0)
        risk[i0:i1, :] = risco_faixa

        # --- Probabilidade de alerta DETER ---
        a_p, b_p  = _beta_params(prob_alerta, concentration=7.0)
        prob[i0:i1, :] = rng.beta(a_p, b_p, size=(ni, m))

        # --- Células bloqueadas (acesso fluvial exclusivo) ---
        mask_fluvial    = rng.random(size=(ni, m)) < frac_fluvial
        blocked[i0:i1, :] = mask_fluvial

        # --- Custo: -1 para bloqueadas, [3,12] para demais ---
        custo_base      = rng.uniform(3.0, 12.0, size=(ni, m))
        cost[i0:i1, :]  = np.where(mask_fluvial, -1.0, custo_base)

    # Garante que origem e destino são sempre acessíveis
    blocked[0, 0]         = False
    blocked[n - 1, m - 1] = False
    cost[0, 0]             = max(float(cost[0, 0]), 1.0)
    cost[n - 1, m - 1]    = max(float(cost[n - 1, m - 1]), 1.0)

    cost_int      = np.where(blocked, -1, np.round(cost).astype(int))
    blocked_count = int(blocked.sum())

    return {
        "risk":    risk,
        "cost":    cost_int,
        "prob":    prob,
        "blocked": blocked,
        "metadata": {
            "cenario":          "C",
            "nome":             "Desmatamento na Amazônia — DETER/PRODES 2020–2024",
            "tamanho":          f"{n}x{m}",
            "blocked_count":    blocked_count,
            "blocked_fraction": round(blocked_count / (n * m), 3),
            "seed":             seed,
            "fontes": [
                "INPE PRODES 2023 — 11.568 km² (terrabrasilis.dpi.inpe.br)",
                "INPE DETER 2020-2024 — alertas por estado",
                "IBGE MUNIC 2022 — municípios com acesso fluvial exclusivo",
            ],
        },
    }


# GRADE PEQUENA PARA TESTES (FORÇA BRUTA, N,M ≤ 5)

def generate_small_grid(n: int, m: int, cenario: str = "A", seed: int = 0) -> dict:
    """
    Gera grade reduzida para testes e validação com Força Bruta.
    Usa os mesmos parâmetros de calibração dos cenários reais.

    Returns:
        dict com 'risk', 'cost', 'prob', 'blocked', 'metadata'
    """
    if cenario == "A":
        return generate_grid_matopiba(n=n, m=m, seed=seed)
    elif cenario == "C":
        return generate_grid_amazonia(n=n, m=m, seed=seed)
    else:
        raise ValueError(f"Cenário '{cenario}' não reconhecido. Use 'A' ou 'C'.")


# SALVAR E CARREGAR GRADES

def save_grids(output_dir: str = None) -> None:
    """
    Gera as grades dos dois cenários e salva em data/processed/.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data" / "processed"

    os.makedirs(output_dir, exist_ok=True)

    geradores = [
        ("A", generate_grid_matopiba),
        ("C", generate_grid_amazonia),
    ]

    for letra, func in geradores:
        print(f"\n{'='*55}")
        print(f"  Cenário {letra} — {func.__doc__.split(chr(10))[1].strip()}")
        print(f"{'='*55}")

        grid   = func(n=GRID_SIZE, m=GRID_SIZE)
        meta   = grid["metadata"]
        prefix = str(output_dir / f"cenario_{letra}_{GRID_SIZE}x{GRID_SIZE}")

        np.save(f"{prefix}_risk.npy",    grid["risk"])
        np.save(f"{prefix}_cost.npy",    grid["cost"])
        np.save(f"{prefix}_prob.npy",    grid["prob"])
        np.save(f"{prefix}_blocked.npy", grid["blocked"])
        with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        r, c, p = grid["risk"], grid["cost"], grid["prob"]
        blk     = grid["blocked"]

        print(f"  Grade:            {meta['tamanho']}")
        print(f"  Risco médio:      {r.mean():.3f} (dp={r.std():.3f})")
        print(f"  Custo médio:      {c[c >= 0].mean():.1f} (excl. bloqueadas)")
        print(f"  Prob. média:      {p.mean():.3f}")
        print(f"  Células bloq.:    {blk.sum()} ({meta.get('blocked_fraction',0)*100:.1f}%)")
        print(f"  Salvo em:         {prefix}_*.npy")


def load_grid(cenario: str, data_dir: str = None) -> dict:
    """
    Carrega uma grade previamente salva de data/processed/.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data" / "processed"

    prefix = str(data_dir / f"cenario_{cenario}_{GRID_SIZE}x{GRID_SIZE}")

    if not os.path.exists(f"{prefix}_cost.npy"):
        raise FileNotFoundError(
            f"Grade do Cenário {cenario} não encontrada.\n"
            f"Execute: python src/data_loader.py"
        )

    with open(f"{prefix}_meta.json", encoding="utf-8") as f:
        metadata = json.load(f)

    return {
        "risk":     np.load(f"{prefix}_risk.npy"),
        "cost":     np.load(f"{prefix}_cost.npy"),
        "prob":     np.load(f"{prefix}_prob.npy"),
        "blocked":  np.load(f"{prefix}_blocked.npy"),
        "metadata": metadata,
    }


# EXECUÇÃO DIRETA

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Global Solution 2026 — Geração das Grades")
    print("="*55)
    save_grids()
    print("\n✅ Todas as grades geradas com sucesso!")
    print("   Próximo passo: python src/brute_force.py\n")