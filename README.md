# Deep Learning e Visão Computacional — Projetos 1 e 2

Rafael de Oliveira Macena Dias
Prof. Allan Segovia Spadini

Repositório com os dois projetos da disciplina, os notebooks reproduzíveis,
os logs do TensorBoard e os checkpoints dos melhores modelos.

| projeto | problema | arquitetura | notebook |
|---|---|---|---|
| **1** | Classificação multiclasse de espécies de pinguim (Palmer Penguins) | MLP densa | `notebooks/projeto1.ipynb` |
| **2** | Classificação binária de radiografia de tórax (NORMAL vs PNEUMONIA) | CNN 2D do zero | `notebooks/projeto2.ipynb` |

---

## Estrutura

```
.
├── notebooks/
│   ├── projeto1.ipynb                         Projeto 1 — MLP (Palmer Penguins)
│   │
│   ├── projeto2.ipynb                         Projeto 2 — CNN, notebook principal (a entrega)
│   ├── projeto2_ablacao_arquiteturas.ipynb    ablação controlada: BatchNorm e tipo de cabeça
│   ├── projeto2_referencia_teto_resnet18.ipynb  ResNet-18 pré-treinada como teto de referência
│   │
│   ├── melhor_modelo.pt                       checkpoint solto da versão inicial do Projeto 1
│   │
│   └── runs/
│       ├── projeto1/                          logs TensorBoard + checkpoints do Projeto 1
│       │   └── checkpoints/                   um .pt por configuração testada (ativação × init × norm)
│       │
│       └── projeto2/
│           ├── INDICE_DE_RUNS.md              mapa dos runs que sustentam o relatório (R1/R2/R3)
│           ├── checkpoints/                   state_dict (.pt) dos melhores modelos
│           └── <run>/                         um diretório por execução
│               ├── events.out.tfevents.*      log do TensorBoard
│               ├── historico.json             perdas, acurácias, AUROC, lr e grad_norm por época
│               ├── metricas_teste.json        métricas finais no conjunto de teste
│               └── figuras/                   curvas, matriz de confusão, ROC, grades de erro
│
├── data/
│   └── penguins_lter.csv                      dataset do Projeto 1 (versionado)
│
├── requirements.txt                           versões exatas (pip)
├── pyproject.toml / uv.lock                   ambiente com uv
└── .python-version                            3.13
```

> O relatório técnico (PDF) e os textos de apoio que sustentaram a redação
> não fazem parte deste repositório — são entregues separadamente.

---

## Ambiente

O projeto usa [uv](https://docs.astral.sh/uv/). Python 3.13.

```bash
# opção A — uv (recomendado, usa o uv.lock)
uv sync
source .venv/bin/activate

# opção B — pip, a partir das versões travadas
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verificação rápida:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Os treinos foram rodados em **Apple Silicon com backend MPS**. Os notebooks
detectam o dispositivo automaticamente e caem para CPU se não houver
acelerador — o resultado numérico não muda, só o tempo.

---

## Datasets

**Projeto 1 — Palmer Penguins.** Já versionado em `data/penguins_lter.csv`.
Nada a baixar.

**Projeto 2 — Chest X-Ray Pneumonia.** Baixado automaticamente pelo
`kagglehub` na primeira execução, para `~/.cache/kagglehub/` (≈2,4 GB):

```python
import kagglehub
caminho = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
```

O notebook guarda o retorno numa variável em vez de escrever o caminho na
mão — é isso que permite rodar em outra máquina sem editar nada. **É o único
"caminho do dataset" do projeto**, e ele se resolve sozinho.

> **Atenção ao nível do diretório.** A raiz do download contém `train/`,
> `val/`, `test/` **e também** uma pasta `chest_xray/` com cópia idêntica dos
> três, mais `__MACOSX/` e `.DS_Store`. Apontar para o nível errado duplica
> todas as contagens sem dar erro nenhum. Os notebooks tratam os dois casos.

Requer credenciais do Kaggle configuradas (`~/.kaggle/kaggle.json` ou as
variáveis `KAGGLE_USERNAME` / `KAGGLE_KEY`).

---

## Como reproduzir

```bash
jupyter lab
```

1. `notebooks/projeto1.ipynb` — roda em segundos, CPU basta
2. `notebooks/projeto2.ipynb` — modelo principal do Projeto 2, é a entrega
3. `notebooks/projeto2_ablacao_arquiteturas.ipynb` — ablação das três variantes de arquitetura
4. `notebooks/projeto2_referencia_teto_resnet18.ipynb` — teto de referência (não é a entrega)

Ou, sem abrir o Jupyter:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 notebooks/projeto2.ipynb
```

Cada notebook roda **do início ao fim sem modificação manual**. O único
parâmetro que se mexe é `HEAD_TYPE` na célula da arquitetura do
`projeto2.ipynb` (`"gap"` ou `"flatten"`), e ele já vem no valor da
configuração final.

### Reprodutibilidade

Com `semear()` cobrindo `random`, `numpy` e `torch` reexecutado no topo do
notebook, o pipeline é determinístico de ponta a ponta:

| verificação | resultado |
|---|---|
| ablação A / B / C, 6 execuções | AUROC idêntica nas seis, até o sexto dígito |
| configuração final, 4 execuções | `train_losses` bit-idênticas nas 30 épocas |
| ResNet-18, 3 execuções | AUROC 0,9625 e matriz idêntica nas três |

**Ressalva honesta:** isso mede *repetibilidade* (mesma seed), não variância
entre seeds. Mudar apenas o estado aleatório move a especificidade em ~3pp —
diferenças menores que isso entre variantes ficam inconclusivas. Detalhes em
`notebooks/runs/projeto2/INDICE_DE_RUNS.md`.

---

## TensorBoard

```bash
tensorboard --logdir notebooks/runs
```

Cobre os treinamentos dos dois projetos, com curvas de loss de treino e
validação, learning rate e métricas de performance por época. No Projeto 2 há
também `grad_norm_media` e `grad_norm_maxima`, usados no diagnóstico de
estabilidade de gradiente.

> **Runs extras.** Além dos runs canônicos documentados em
> `INDICE_DE_RUNS.md`, a pasta `notebooks/runs/projeto2/` acumula diretórios
> de reexecuções exploratórias posteriores (algumas sem `metricas_teste.json`,
> por terem sido interrompidas antes da avaliação final). Eles não invalidam
> os números do relatório — apenas não são a fonte deles. Para saber qual run
> sustenta qual número, use sempre o índice, não a data mais recente.

---

## Checkpoints

Em `notebooks/runs/projeto2/checkpoints/`:

| arquivo | modelo |
|---|---|
| `melhor_modelo_gap_agrupado_com_peso_smooth_20260827-230807.pt` | **configuração final (R1, a entrega)** — CNN com cabeça `AdaptiveAvgPool`, class weights e label smoothing |
| `melhor_modelo_gap_agrupado_com_peso_smooth.pt` | mesma configuração, checkpoint de uma reexecução posterior sem timestamp no nome |
| `melhor_modelo_resnet18_referencia_teto.pt` | ResNet-18 pré-treinada (referência de teto, **não é a entrega**) |

Formato do `.pt`: dicionário com as chaves `model` (o `state_dict`), `epoca`,
`val_loss`, `val_acc` e `val_auroc`.

```python
estado = torch.load(caminho, map_location="cpu")
modelo.load_state_dict(estado["model"])
```

Em `notebooks/runs/projeto1/checkpoints/` estão os `.pt` de cada configuração
testada nos experimentos de ativação/inicialização/normalização do Projeto 1
(`model_<ativação>_<inicialização>.pt`, `model_Com_BatchNorm1d.pt`, etc.), mais
os dos experimentos de gradiente explodindo/desaparecendo com e sem correção.

### O que não está versionado

Cada run de ablação salva mais três `.pt` dentro do próprio diretório
(`ablacao_*/A_sem_norm.pt`, `B_plantvillage.pt`, `C_gap.pt`) — ignorados pelo
Git (ver `.gitignore`). São reproduzíveis executando o notebook de ablação.

---

## Resultados — resumo

Conjunto de teste do Projeto 2: 234 NORMAL + 390 PNEUMONIA.

| modelo | acc | sens | spec | bal | AUROC |
|---|---|---|---|---|---|
| **CNN, cabeça `gap` + weight + smoothing (R1, entrega)** | **84,5%** | **98,2%** | **61,5%** | **79,9%** | **0,9277** |
| ResNet-18 pré-treinada (R2, teto, não é a entrega) | 85,9% | 99,5% | 63,2% | — | 0,9625 |

Ablação de arquitetura (R3, mesma espinha convolucional, variando uma coisa
por vez):

| arquitetura | parâmetros | melhor época | AUROC teste | especificidade |
|---|---|---|---|---|
| `A_sem_norm` | 8.482.626 | 10 | 0,9334 | 43,6% |
| `B_plantvillage` (+BatchNorm) | 8.483.074 | 18 | 0,9381 | 37,6% |
| `C_gap` (cabeça GAP) | 93.954 | 6 | 0,9199 | 50,0% |

O detalhamento completo — matriz de confusão, threshold escolhido na
validação, AUPRC, delta val−teste e a comparação de gradiente entre cabeças —
está em `notebooks/runs/projeto2/INDICE_DE_RUNS.md`.
