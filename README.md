# Global Solution 2026 — Otimização de Rotas de Resposta a Riscos Agroclimáticos

**FIAP — Engenharia de Software | 4º Semestre**
**Disciplina:** Algoritmos e Estruturas de Dados Avançados
**Professor:** André Marques

---

## Integrantes

| Nome | RM |
|---|---|
| Luis Filipe Crivellaro | 560877 |
| Felipe Silva do Prado Lima | 559848 |
| Rafael Mandel | 560333 |

---

## Descrição do Projeto

Sistema de otimização de rotas de resposta a riscos agroclimáticos que combina
dados orbitais reais (NASA/INPE/INMET) com três abordagens algorítmicas aplicadas
a dois cenários brasileiros críticos:

| Cenário | Região | Grade | Dados |
|---|---|---|---|
| **A — Seca no Cerrado/Nordeste** | MATOPIBA (MA, TO, PI, BA) | 50×50 | NDVI MODIS + INMET |
| **C — Desmatamento na Amazônia** | Arco do desmatamento | 50×50 | INPE PRODES/DETER 2020–2024 |

**ODS relacionados:** 2 · 8 · 9 · 11 · 13

---

## Estrutura do Repositório

```
global-solution-2026/
├── README.md                        # Este arquivo
├── requirements.txt                 # Dependências Python
├── data/
│   ├── raw/                         # Dados brutos (baixar conforme fontes abaixo)
│   └── processed/                   # Grades N×M geradas (criadas por data_loader.py)
│       ├── cenario_A_50x50_risk.npy
│       ├── cenario_A_50x50_cost.npy
│       ├── cenario_A_50x50_prob.npy
│       ├── cenario_A_50x50_blocked.npy
│       ├── cenario_A_50x50_meta.json
│       └── cenario_C_50x50_*.npy / *.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Geração das grades calibradas por dados reais
│   ├── brute_force.py               # Enumeração completa — oráculo de validação (N,M ≤ 5)
│   ├── dynamic_programming.py       # DP bottom-up + memoização + reconstrução de caminho
│   ├── monte_carlo.py               # Simulação estocástica K=10.000 cenários (Beta)
│   ├── performance_monitor.py       # Tempo, memória, iterações e curvas de escalabilidade
│   ├── visualizations.py           # Geração centralizada das 5 figuras obrigatórias
│   └── analise_resultados.ipynb     # Análise interativa + escala de decisão (4 níveis)
├── tests/
│   └── test_algorithms.py           # 42 testes automatizados (pytest)
└── report/
    ├── relatorio_final.pdf          # Relatório técnico (4 páginas)
    ├── dp_heatmap_cenario_A.png     # Figura 1a — Heatmap DP Cenário A
    ├── dp_heatmap_cenario_C.png     # Figura 1b — Heatmap DP Cenário C
    ├── mc_distribuicao_cenario_A.png# Figura 2a — MC Cenário A
    ├── mc_distribuicao_cenario_C.png# Figura 2b — MC Cenário C
    ├── escalabilidade.png           # Figura 3 — Curva de escalabilidade
    ├── scatter_custo_tempo.png      # Figura 4 — Trade-off custo × tempo
    ├── sensibilidade_comparativa.png# Figura 5 — Análise de sensibilidade ±20%
    ├── fb_crescimento_caminhos.png  # Crescimento exponencial da Força Bruta
    └── escala_decisao.png           # Escala de decisão (4 níveis)
```

---

## Como Executar

### 1. Clonar e instalar dependências

```bash
git clone https://github.com/rflMandell/GS-DP-2026.git
cd GS-DP-2026

python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Gerar as grades dos cenários

```bash
python src/data_loader.py
```

> Cria os arquivos `.npy` em `data/processed/` com as grades 50×50
> calibradas pelos dados reais de INPE, INMET e MODIS.

### 3. Executar os algoritmos individualmente

```bash
# Força Bruta — instâncias pequenas N,M ≤ 5 (oráculo de validação)
python src/brute_force.py

# Programação Dinâmica — bottom-up + memoização + heatmaps
python src/dynamic_programming.py

# Monte Carlo — K=10.000 cenários + análise de sensibilidade ±20%
python src/monte_carlo.py
```

### 4. Monitor de desempenho

```bash
python src/performance_monitor.py
```

> Gera tabela comparativa e curvas de escalabilidade para
> N = 3, 5, 10, 20, 50, 100 (mediana de 5 execuções por ponto).

### 5. Gerar todas as figuras obrigatórias

```bash
python src/visualizations.py
```

> Gera as 7 figuras em `report/` incluindo as 5 obrigatórias do enunciado.

### 6. Notebook de análise interativa

```bash
jupyter notebook src/analise_resultados.ipynb
```

> Execute as células em ordem com `Shift+Enter`. O notebook produz
> todas as figuras e a escala de decisão de 4 níveis interativamente.

---

## Resultados Principais

| Métrica | Cenário A (MATOPIBA) | Cenário C (Amazônia) |
|---|---|---|
| Custo ótimo DP | **546.31** | **969.94** |
| Custo MC médio (μ) | 415.88 ± 2.06 | 851.83 ± 5.35 |
| IC 95% | [411.77, 419.88] | [841.25, 862.04] |
| Impacto +20% prob. | +2.91% no custo | +3.80% no custo |
| Células bloqueadas | 0% | 33.7% |
| Testes automatizados | **42/42** | — |

---

## Descrição dos Módulos

### `src/data_loader.py`
Gera as grades N×M calibradas por valores base reais documentados
(NDVI MODIS, precipitação INMET, desmatamento PRODES). Usa distribuições
Beta para amostrar os atributos `r[i][j]`, `c[i][j]` e `p[i][j]`
por sub-região. No Cenário C, inclui corredor diagonal representando
a BR-163/BR-230 para garantir viabilidade da grade.

### `src/brute_force.py`
Enumeração recursiva completa de todos os caminhos válidos (N,M ≤ 5).
Contador de chamadas instrumentado. Gráfico de crescimento combinatório.
**Serve como oráculo**: para toda instância pequena, FB e DP devem coincidir.

### `src/dynamic_programming.py`
Duas variações da DP 2D com recorrência
`dp[i][j] = cost[i][j]×(1+risk[i][j]) + min(dp[i-1][j], dp[i][j-1])`:
- **Bottom-Up**: tabulação iterativa O(N×M), sem overhead de pilha
- **Memoização**: recursão com `lru_cache`, overhead visível em N≥50

Reconstrução do caminho ótimo por backtracking. Suporte a células
bloqueadas (`cost=-1`). Gera heatmaps com caminho destacado.

### `src/monte_carlo.py`
Simula K=10.000 cenários amostrando `p'[i][j] ~ Beta(α,β)` calibrada.
DP vetorizada via numpy (batches) para eficiência: K=10.000 em <3s.
Calcula média, mediana, desvio padrão e IC 95%. Análise de sensibilidade
com p[i][j] ±20%. Gera histograma + boxplot obrigatórios.

### `src/performance_monitor.py`
Instrumenta os 3 algoritmos para N = 3, 5, 10, 20, 50, 100 medindo
tempo (`time.perf_counter`), memória (`tracemalloc`) e iterações.
Mediana de 5 repetições por ponto para eliminar outliers de SO.
Gera curva de escalabilidade e scatter custo×tempo.

### `src/visualizations.py`
Ponto único de entrada para gerar todas as 5 figuras obrigatórias.
Importa funções de plotagem dos módulos anteriores e adiciona a
figura de análise de sensibilidade comparativa (Figura 5).

---

## Fontes de Dados

| Fonte | Dado | Cenário |
|---|---|---|
| [NASA Earthdata — MODIS MOD13A3](https://earthdata.nasa.gov) | NDVI histórico 2000–2024 | A |
| [INMET BDMEP](https://bdmep.inmet.gov.br) | Normais climatológicas 1991–2020 | A |
| [IBGE/PNUD](https://ibge.gov.br) | Atlas Desenvolvimento Humano Municipal 2021 | A |
| [INPE PRODES 2023](https://terrabrasilis.dpi.inpe.br) | 11.568 km² desmatados | C |
| [INPE DETER 2020–2024](https://terrabrasilis.dpi.inpe.br) | Alertas por estado | C |
| [IBGE MUNIC 2022](https://ibge.gov.br) | Municípios com acesso fluvial exclusivo | C |

> **Abordagem de dados:** valores base reais por região usados como
> parâmetros de distribuições Beta que geram as grades sintéticas calibradas.
> Documentado na Seção 5 do enunciado (Observação sobre dados sintéticos).

---

## 📚 Referências

- ESA Earth Observation: [esa.int](https://esa.int/Applications/Observing_the_Earth)
- NASA Earthdata: [earthdata.nasa.gov](https://earthdata.nasa.gov)