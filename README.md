# Guia — Projeto 1: Pipeline MLP (Palmer Penguins)

Checklist de passos, na ordem que faz sentido implementar. Cada item é
algo pra você pesquisar e implementar — não tem código aqui, só o
"o que" e o "por quê", pra você decidir o "como".

Dataset: Kaggle `parulpandey/palmer-archipelago-antarctica-penguin-data`
("Palmer Archipelago (Antarctica) Penguin Data", 344 linhas, 7 colunas).
- Classificação: alvo `species` (Adelie / Chinstrap / Gentoo — 3 classes, ~152/68/124, moderadamente desbalanceado)
- Projeto 1 é só classificação — sem variante de regressão (confirmado pelo enunciado real do projeto)

---

## 0. Ambiente
- [X] Confirmar que PyTorch, Jupyter e TensorBoard rodam antes de treinar qualquer coisa (import + versão + `torch.cuda.is_available()`)
- [X] Ambiente do projeto já criado com `uv` — só validar que abre no Jupyter

## 1. Dados
- [x] Baixar o dataset (`kagglehub` já está nas dependências)
- [x] Explorar: tipos de coluna, valores faltantes reais (`culmen_length_mm`, `culmen_depth_mm`, `flipper_length_mm`, `body_mass_g` têm 2 NaN cada; `sex` tem 10 NaN — diferente do caso do Pokémon, aqui NaN é medição ausente de verdade, não uma categoria válida), distribuição de `species` (desbalanceamento mais leve, mas ainda existe)
- [x] Definir pré-processamento: o que fazer com as linhas de NaN (dropar vs imputar — cheque se o NaN aparece na linha inteira ou só em colunas isoladas), encoding de variáveis categóricas (`island`, `sex`), normalização/padronização das medidas numéricas
- [x] Split treino/validação/teste (pesquisar: por que não pode usar o mesmo split pra tudo, e por que as métricas finais têm que vir do teste, não da validação)

## 2. MLP do zero (classificação)
- [X] Implementar com `nn.Module` puro — sem Lightning nem outras abstrações de alto nível
- [X] Definir explicitamente cada camada, ativação e o `forward`
- [X] Justificar a escolha de profundidade e largura em função do problema (dataset pequeno, poucas features → cuidado com over-parametrização)
- [X] Testar pelo menos 2 funções de ativação diferentes (ex: ReLU vs Tanh, ou ReLU vs LeakyReLU) e comparar curvas de loss
- [X] Comparar inicialização de pesos: Xavier ou He vs a inicialização padrão do PyTorch — documentar o efeito

## 3. Training loop
- [X] `DataLoader` para treino e validação
- [X] Loop com validação por época (loss + métrica em cada época)
- [X] Checkpointing via `state_dict`: salvar o melhor modelo, saber recarregar e reproduzir predições

## 4. Estabilidade do treinamento (tratar como requisito, não bônus)
- [X] Calcular e plotar gradient norms por época/step — diagnosticar vanishing ou exploding gradients
- [X] Implementar BatchNorm ou LayerNorm e comparar o efeito nas curvas de loss
- [X] Aplicar Dropout e verificar o impacto no gap treino/validação
- [X] Trocar otimizador padrão por Adam ou AdamW + learning rate scheduling, monitorado no TensorBoard

## 5. Relatório
- [X] Definir e documentar um baseline não trivial (ex: classificador majoritário) pra comparação
- [X] Analisar curvas de loss nomeando o fenômeno observado: overfitting, underfitting, instabilidade ou convergência
- [X] Escolher pelo menos 1 problema real encontrado nos experimentos e documentar como:
  1. Hipótese formulada **antes** da correção
  2. Correção aplicada
  3. Efeito medido (gradient norm, spike de loss, distribuição de ativações — evidência objetiva, não achismo)
- [X] Interpretar a matriz de confusão da classificação: que padrões de erro aparecem no seu dado específico e por quê (ex: confunde Adelie com Chinstrap, as duas espécies morfologicamente mais próximas, enquanto Gentoo se separa bem?)

---

## Estrutura do repo
```
projeto-redes-neurais/
├── data/                      # dataset baixado (gitignored)
├── notebooks/
│   └── runs/projeto1/         # tudo gerado pelo notebook (gitignored)
│       ├── checkpoints/       # state_dict (.pt) salvos
│       └── <run>/             # logs do TensorBoard, um por execução
├── GUIA.md
└── pyproject.toml             # uv, já com torch/tensorboard/sklearn/etc
```
