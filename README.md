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

Sistema de otimização de rotas de resposta a riscos agroclimáticos, combinando dados orbitais reais (NASA/INPE/INMET) com três abordagens algorítmicas — Força Bruta, Programação Dinâmica 2D e Simulação Monte Carlo — aplicadas a dois cenários brasileiros críticos:

- **Cenário — Seca no Cerrado/Nordeste (MATOPIBA):** grade 50×50 representando municípios das regiões Maranhão, Tocantins, Piauí e Bahia, com dados NDVI/MODIS e precipitação INMET.
- **Cenário — Desmatamento na Amazônia:** grade derivada dos alertas DETER/PRODES do INPE (2020–2024), com células de acesso fluvial exclusivo modeladas como bloqueadas.

**ODS relacionados:** 2 · 8 · 9 · 11 · 13

---

## Estrutura do Repositório

```
global-solution-2026/
├── README.md                        # Este arquivo
├── requirements.txt                 # Dependências Python
├── data/
│   ├── raw/                         # Dados brutos (NDVI, pluviometria, DETER...)
│   └── processed/                   # Grades N×M geradas para cada cenário
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Coleta e processamento dos dados reais
│   ├── brute_force.py               # Enumeração completa — baseline de validação
│   ├── dynamic_programming.py       # Tabulação DP + memoização + reconstrução
│   ├── monte_carlo.py               # Simulação estocástica K>=10.000 cenários
│   ├── performance_monitor.py       # Tempo, memória, iterações e curvas
│   └── visualizations.py            # Heatmap DP, histograma MC, gráficos
├── notebooks/
│   └── analise_resultados.ipynb     # Análise interativa e escala de decisão
├── tests/
│   └── test_algorithms.py           # Testes unitários e validação FB vs DP
└── report/
    └── relatorio_final.pdf          # Relatório técnico (máx. 4 páginas)
```

---

## Como Executar

### 1. Clonar o repositório
```bash
git clone https://github.com/rflMandell/GS-DP-2026.git
cd GS-DP-2026
```

### 2. Criar ambiente virtual e instalar dependências
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Gerar as grades dos cenários (dados processados)
```bash
python src/data_loader.py
```

### 4. Executar os algoritmos
```bash
# Força Bruta (instâncias pequenas N,M ≤ 5)
python src/brute_force.py

# Programação Dinâmica (todos os tamanhos)
python src/dynamic_programming.py

# Monte Carlo (K=10.000 cenários)
python src/monte_carlo.py
```

### 5. Monitoramento de desempenho
```bash
python src/performance_monitor.py
```

### 6. Gerar todas as visualizações
```bash
python src/visualizations.py
```

### 7. Executar testes automatizados
```bash
pytest tests/ -v
```

### 8. Abrir notebook de análise
```bash
jupyter notebook notebooks/analise_resultados.ipynb
```

---

## Fontes de Dados

| Fonte | Dado |
|---|---|
| [NASA Earthdata — MODIS](https://earthdata.nasa.gov) | NDVI histórico 2000–2024 |
| [INMET BDMEP](https://bdmep.inmet.gov.br) | Precipitação histórica |
| [INPE PRODES/DETER](https://terrabrasilis.dpi.inpe.br) | Alertas de desmatamento 2020–2024 |
| [IBGE](https://ibge.gov.br) | Limites municipais e dados socioeconômicos |

---

## Referências

- ESA Earth Observation: [esa.int](https://esa.int/Applications/Observing_the_Earth)
- NASA Earthdata: [earthdata.nasa.gov](https://earthdata.nasa.gov)
