# PME 3361 — Transferência de Calor: Projeto I (Grupo 15C)

Avaliação numérica e otimização de processo de colagem térmica de película plástica por laser em chapa de aço — **Escola Politécnica da Universidade de São Paulo (EPUSP)**.

- **Grupo:** `15C`
- **Entrega:** `01/12/2026`
- **Integrantes:**
  - Gabriel Barros Louro Alves — NUSP `15586270`
  - Henrique Hideki Miashiro — NUSP `13672209`
  - Lucas Augusto Gazotti — NUSP `15471223`
  - Guilherme Watanabe Miyake — NUSP `14576555`

---

## 1. Visão Geral do Problema

Uma película polimérica de largura $w_1 = 44\text{ mm}$ é colada na região central de uma chapa de aço de espessura $d = 1{,}25\text{ mm}$, largura total $w_2 = 132\text{ mm}$ ($p = w_2/w_1 = 3$) e comprimento unitário ($1\text{ m}$). Um feixe de laser aplica um fluxo de calor superficial $q''$ durante $\Delta t_L = 10\text{ s}$, seguido de resfriamento até $t_{\text{final}} = 30\text{ s}$, com convecção em ambas as faces da chapa.

### Parâmetros Fixos e do Grupo 15C

| Parâmetro | Símbolo | Valor |
| :--- | :---: | :---: |
| Massa específica do aço | $\rho$ | $7850\text{ kg/m}^3$ |
| Calor específico do aço | $c_p$ | $435\text{ J/(kg}\cdot\text{K)}$ |
| Condutividade térmica do aço | $k$ | $60\text{ W/(m}\cdot\text{K)}$ |
| Difusividade térmica | $\alpha = k/(\rho c_p)$ | $1{,}7571 \times 10^{-5}\text{ m}^2/\text{s}$ |
| Espessura / Largura película / Largura chapa | $d$ / $w_1$ / $w_2$ | $1{,}25\text{ mm}$ / $44\text{ mm}$ / $132\text{ mm}$ |
| Coeficiente de convecção (**Grupo 15C**) | $h$ | $150\text{ W/(m}^2\cdot\text{K)}$ |
| Fluxo de calor do laser (**Grupo 15C**) | $q''$ | $80000\text{ W/m}^2$ |
| Temperatura ambiente (**Grupo 15C**) | $T_\infty$ | $25\,^\circ\text{C}$ |
| Critério de colagem satisfatória ($|x| \le w_1/2$, $t \le 10\text{ s}$) | $[T_{\min}, T_{\max}]$ | $90\,^\circ\text{C} \le T \le 200\,^\circ\text{C}$ |

---

## 2. Modelagem Matemática e Método Numérico

1. **Validade da hipótese 1D na espessura:** o número de Biot transversal é $\text{Bi}_y = h d / k = 3{,}125 \times 10^{-3} \ll 0{,}1$, permitindo desprezar gradientes térmicos na espessura e explorar a simetria na meia chapa $0 \le x \le L = w_2/2 = 66\text{ mm}$.
2. **Adimensionalização (notas de aula do Prof. Ernani V. Volpe):**
   $$\Theta = \frac{T - T_\infty}{T_\infty}, \qquad \tilde{x} = \frac{x}{L}, \qquad \tilde{t} = \frac{\alpha t}{L^2}, \qquad \text{Bi} = \frac{h L^2}{k d}, \qquad \tilde{q} = \frac{q'' L^2}{k d T_\infty}$$
   Para o Grupo 15C: $\text{Bi} = 8{,}712$, $\tilde{q} = 185{,}856$, $\tilde{t}_L = 0{,}040337$ ($10\text{ s}$) e $\tilde{t}_{\text{final}} = 0{,}121011$ ($30\text{ s}$).
3. **Discretização por Diferenças Finitas:**
   - Malha base com $N = 200$ nós internos ($\Delta\tilde{x} = 1/201$, onde o nó $j = 67$ coincide exatamente com a borda da película $x = 22{,}000\text{ mm}$) e $N_t = 3000$ passos temporais.
   - Derivada segunda de 2ª ordem nos nós internos ($j = 2, \dots, N$) e aproximação de Lomax & Pulliam (p. 44) na fronteira de simetria ($j = 1$).
   - Integração temporal por **Euler Implícito**, montando a matriz constante $\mathbf{M} = (1 + 2\,\text{Bi}\,\Delta\tilde{t})\mathbf{I} - \Delta\tilde{t}\,\mathbf{A}$ com **fatoração LU esparsa única** fora do laço temporal (`scipy.sparse.linalg.splu` em Python e `lu` em MATLAB/Octave).
   - Reconstrução de 2ª ordem no centro ($\Theta_0 = (4\Theta_1 - \Theta_2)/3$) e condição de Dirichlet $\Theta_{N+1} = 0$ na extremidade.

---

## 3. Estrutura do Repositório

```text
Transcal/
├── README.md                              # Documentação geral do trabalho e do código
├── src/
│   ├── simulacao.py                       # Solver principal em Python, validações, soluções e figuras
│   ├── simulacao.m                        # Versão espelho compatível com MATLAB e GNU Octave
│   └── resultados_15C.json                # Métricas numéricas exatas exportadas pela simulação
└── overleaf/                              # Projeto LaTeX pronto para compilação no Overleaf
    ├── main.tex                           # Relatório técnico completo em LaTeX (EPUSP)
    ├── simulacao.py                       # Cópia local do código incluída no Apêndice A
    └── figuras/                           # Figuras em 300 DPI (.pdf e .png)
        ├── fig_superficie_3d.{pdf,png}
        ├── fig_perfis_espaciais.{pdf,png}
        ├── fig_evolucao_temporal.{pdf,png}
        └── fig_comparacao_solucoes.{pdf,png}
```

---

## 4. Explicação do Código

### Script Python ([src/simulacao.py](file:///home/gabrielbla/Projetos/Simulaçoes/Transcal/src/simulacao.py))

Organizado de forma modular em torno de funções independentes:

- `resolver_placa(h, q_flux, T_inf, N=200, Nt=3000, t_final=30.0, dt_laser=10.0, w_laser=W1, T_init=None)`:
  Constrói a matriz tridiagonal esparsa `A` (`scipy.sparse.diags`) e a matriz do sistema implícito `M`, realiza uma única fatoração LU (`splu`), aplica a máscara espacial do feixe (`x_int < w_laser / W2 - 1e-12`) e resolve o transiente aceitando tanto fluxo constante quanto perfil variável no tempo `q_flux(t)` (usado no pulso em dois estágios) e temperatura inicial arbitrária `T_init` (usada no pré-aquecimento).
- `calcular_metricas(res, w1=W1, dt_laser=10.0, T_lo=90.0, T_hi=200.0)`:
  Extrai $T_{\max}$ no pulso, $T(0, 10\text{ s})$, $T(11\text{ mm}, 10\text{ s})$, $T(22\text{ mm}, 10\text{ s})$, $T(44\text{ mm}, 10\text{ s})$, gradiente $\Delta T_{\text{pel}}$, tempos de cruzamento de $90\,^\circ\text{C}$ ($t_{90,\text{centro}}$ e $t_{90,\text{borda}}$) por interpolação linear, duração útil dentro da janela $[90, 200]\,^\circ\text{C}$ e margem térmica abaixo de $200\,^\circ\text{C}$.
- `validar_benchmark_11A()`:
  Executa teste automatizado com asserções rigorosas para o Grupo 11A antes de prosseguir para o Grupo 15C.
- `estudo_convergencia_malha(h, q_flux, T_inf)`:
  Avalia independência espacial ($N \in \{100, 200, 400\}$ com $N_t = 3000$) e temporal ($N_t \in \{1500, 3000, 6000\}$ com $N = 200$).
- `simular_solucoes_engenharia(h, q_base, T_inf)`:
  Realiza varreduras paramétricas de fluxo $q''$ e largura do feixe $w_{\text{laser}}$ e compara formalmente as soluções selecionadas (**A**, **B**, **C**, **D** e referência de carenagem com $h = 100\text{ W/(m}^2\cdot\text{K)}$).
- `gerar_figuras(res_base, met_base, solucoes, dir_fig)`:
  Gera os quatro gráficos de alta resolução (300 DPI) em `.pdf` e `.png` diretamente em `overleaf/figuras/`.
- `main()`:
  Orquestra validação, caso base, convergência, soluções, geração de figuras e exportação de `src/resultados_15C.json`.

### Versão MATLAB / GNU Octave ([src/simulacao.m](file:///home/gabrielbla/Projetos/Simulaçoes/Transcal/src/simulacao.m))

Implementada como função principal `simulacao()` com funções locais (`resolver_placa`, `calcular_metricas`, `tempo_cruzamento`, `duracao_janela`, `imprimir_resumo`), utilizando `spdiags`, `speye`, `[L_lu, U_lu, P_lu, Q_lu] = lu(M)` e `interp1`. Reproduz exatamente os resultados numéricos da versão Python e inclui bloco de plotagem 3D opcional controlado por `gerar_figuras = false`.

---

## 5. Como Executar

### Python

Com `uv` (ambiente isolado, inclusive offline se o cache estiver disponível):

```bash
uv run --with numpy --with scipy --with matplotlib python src/simulacao.py
```

Ou com `python3` / `venv` padrão (requer `numpy`, `scipy` e `matplotlib`):

```bash
python3 src/simulacao.py
```

### MATLAB ou GNU Octave

- **GNU Octave (CLI):**
  ```bash
  octave-cli --no-gui --eval "addpath('src'); simulacao()"
  ```
- **MATLAB:** adicione `src/` ao *path* (ou abra a pasta `src/`) e execute `simulacao`.

### Relatório LaTeX no Overleaf

1. Compacte o conteúdo de `overleaf/` (`main.tex`, `simulacao.py` e `figuras/`) em um `.zip` (ou envie os arquivos diretamente para um novo projeto no Overleaf).
2. Utilize o compilador padrão **pdfLaTeX**.

---

## 6. Resumo dos Principais Resultados

- **Benchmark Grupo 11A ($N=200, N_t=3000$):** $\text{Bi} = 5{,}808000$, $\tilde{q} = 174{,}240000$, $T_{\max} = 152{,}23\,^\circ\text{C}$, $t_{90,\text{borda}} = 9{,}30\text{ s}$ e $T_{\text{borda}}(10\text{ s}) = 93{,}83\,^\circ\text{C}$ — **aprovado**.
- **Convergência de malha (Grupo 15C):** erros máximos de $0{,}96\%$ na malha espacial ($N=200$ vs. $N=400$) e $0{,}012\%$ na malha temporal ($N_t=3000$ vs. $N_t=6000$).
- **Caso Base vs. Soluções de Engenharia (Grupo 15C):**

| Cenário | $T_{\max}$ [°C] | $T_{\text{borda}}(10\text{ s})$ [°C] | $\Delta T_{\text{pel}}$ [°C] | $t_{90,\text{borda}}$ [s] | Janela útil [s] | Margem $< 200\,^\circ\text{C}$ [°C] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Caso base ($80\text{ kW/m}^2$, $44\text{ mm}$) | $147{,}80$ | $91{,}02$ | $56{,}78$ | $9{,}78$ | $0{,}224$ | $52{,}20$ |
| Sol. A ($q'' = 105\text{ kW/m}^2$, $44\text{ mm}$) | $186{,}18$ | $111{,}65$ | $74{,}52$ | $6{,}77$ | $3{,}23$ | $13{,}82$ |
| Sol. B ($w_{\text{laser}} = 60\text{ mm}$, $80\text{ kW/m}^2$) | $155{,}76$ | $129{,}70$ | $26{,}06$ | $4{,}88$ | $5{,}12$ | $44{,}24$ |
| **Sol. C (2 estágios $180 \to 75\text{ kW/m}^2$ + $60\text{ mm}$)** | **$171{,}37$** | **$140{,}76$** | **$30{,}61$** | **$1{,}76$** | **$8{,}24$** | **$28{,}63$** |
| Sol. D (Pré-aquecimento $T_0 = 55\,^\circ\text{C}$) | $162{,}65$ | $105{,}60$ | $57{,}05$ | $6{,}09$ | $3{,}91$ | $37{,}35$ |

- **Conclusão técnica:** o caso base permanece apenas **$0{,}224\text{ s}$** com toda a película acima de $90\,^\circ\text{C}$ (margem de apenas $1{,}02\,^\circ\text{C}$ na borda). A **Solução C** é recomendada por elevar o tempo útil em **36,7×** ($8{,}24\text{ s}$ no pulso de $10\text{ s}$) com folga térmica segura ($28{,}63\,^\circ\text{C}$ abaixo de $200\,^\circ\text{C}$), enquanto a **Solução B** destaca-se como alternativa passiva simples ($5{,}12\text{ s}$ úteis e menor gradiente espacial, $26{,}06\,^\circ\text{C}$).
