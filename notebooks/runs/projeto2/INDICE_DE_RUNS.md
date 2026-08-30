# Índice de runs — Projeto 2

Três runs sustentam todo o relatório, um por notebook. Cada um é a execução
mais recente e completa do seu notebook, e os prints do relatório saem deles.

Runs anteriores foram removidos: eram réplicas da mesma configuração ou
execuções de versões do código que não existem mais. Manter dezenas de
diretórios equivalentes não acrescenta evidência e dificulta auditar qual
número veio de onde.

---

## R1 — `gap_agrupado_com_peso_smooth_20260827-230807`

**Notebook:** `projeto2.ipynb` · **É a entrega.**

Cabeça `AdaptiveAvgPool2d`, split agrupado por paciente, `CrossEntropyLoss`
com pesos de classe e `label_smoothing=0.05`, 128×128, 30 épocas com early
stopping de paciência 7.

| métrica | valor |
|---|---|
| checkpoint | época 27 |
| matriz (TN/FP/FN/TP) | 144 / 90 / 7 / 383 |
| acurácia | 84.5% |
| sensibilidade | 98.2% |
| especificidade | 61.5% |
| acurácia balanceada | 79.9% |
| AUROC teste | 0.9277 |
| AUPRC teste | 0.9385 |
| AUROC validação | 0.9969 |
| delta val−teste | +0.0692 |
| threshold escolhido na validação | 0.680 |
| — sensibilidade no teste | 96.2% |
| — especificidade no teste | 74.4% |

**Artefatos:** `historico.json` (perdas, acurácias, AUROC, lr e norma de
gradiente por época), `metricas_teste.json`, `comparacao_gradiente_cabecas.json`,
`figuras/` e o log do TensorBoard.

### Comparação de gradiente entre as cabeças

Medida pela célula 15, 5 épocas, seed 42, tudo idêntico menos o
classificador:

| | flatten | gap |
|---|---|---|
| parâmetros | 8,483,074 | 93,954 |
| grad médio ep1 | 17.011 | 2.116 |
| grad máximo ep1 | 189.171 | 15.342 |
| razão pico/mediana | 44.9× | 7.3× |

---

## R2 — `resnet18_referencia_teto_20260828-001438`

**Notebook:** `projeto2_referencia_teto_resnet18.ipynb` · **Referência de teto,
não é a entrega.**

ResNet-18 pré-treinada, fine-tuning completo, AdamW com lr 3e-4, 15 épocas.
Serve para quantificar quanto do erro residual é atacável por arquitetura.

| métrica | valor |
|---|---|
| matriz (TN/FP/FN/TP) | 148 / 86 / 2 / 388 |
| acurácia | 85.9% |
| sensibilidade | 99.5% |
| especificidade | 63.2% |
| AUROC teste | 0.9625 |
| delta val−teste | +0.0369 |

---

## R3 — `ablacao_20260828-094233`

**Notebook:** `projeto2_ablacao_arquiteturas.ipynb`

Três arquiteturas com a mesma espinha convolucional, variando **uma** coisa
por comparação. A → B isola o BatchNorm; B → C isola a cabeça. `set_seed()` é
chamada antes de cada uma, para que a diferença medida não misture design com
estado aleatório.

| arquitetura | params | melhor época | AUROC teste | especificidade |
|---|---|---|---|---|
| `A_sem_norm` | 8,482,626 | 10 | 0.9334 | 43.6% |
| `B_plantvillage` | 8,483,074 | 18 | 0.9381 | 37.6% |
| `C_gap` | 93,954 | 6 | 0.9199 | 50.0% |

**Artefatos:** `resultados.json` com histórico e varredura de threshold das
três, um `.pt` por arquitetura, e `figuras/` com os painéis comparativos.

---

## Reprodutibilidade

| configuração | execuções | resultado |
|---|---|---|
| ablação A / B / C | 6 | AUROC idêntica nas seis, até o sexto dígito |
| configuração final | 4 | `train_losses` bit-idênticas nas 30 épocas |
| ResNet-18 | 3 | AUROC 0.9625 e matriz idêntica nas três |

O determinismo vem de `semear()` cobrindo `random`, `numpy` e `torch`, mais
`random_state=42` explícito no `StratifiedGroupKFold`.

Isso mede **repetibilidade**, não variância entre seeds. Cada configuração foi
executada com uma seed só. Como mudar apenas o estado aleatório move a
especificidade cerca de 3pp, diferenças menores que isso entre variantes ficam
inconclusivas.
