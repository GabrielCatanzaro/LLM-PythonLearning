# CP2/CP3 - Generative AI for Engineering
## Tutor de Python para iniciantes: fine-tuning de uma LLM local em GPU AMD (RX 6600)

**Grupo:** 

Gabriel Gomes Catanzaro / RM93445 
Lucas Gomes Alcântara / RM98766 
Henrique Canali Cuzato / RM551326 
Filipe Prado Menezes / RM98765 
Cássio Eid Kobaysahi Yonetsuka / RM99678
Pedro Henrique Salvitti Habiro / RM88166

## 1. Problema escolhido
Alunos iniciantes em Python fazem perguntas simples e repetitivas (sintaxe, erros comuns, funcoes embutidas).
O objetivo e ajustar uma LLM pequena, que roda localmente, para responder **em portugues, de forma curta e com exemplo de codigo**,
no formato de um tutor.

## 2. Modelo e tecnica
| Item | Escolha |
|---|---|
| Modelo base | `Qwen/Qwen2.5-0.5B-Instruct` |
| Tecnica | Fine-tuning **LoRA** (PEFT) nas camadas de atencao `q_proj, k_proj, v_proj, o_proj` |
| Dados | `data/dataset.jsonl` - 67 pares pergunta/resposta (gerados por `criar_dataset.py`) |
| Precisao | float32 (mais estavel no DirectML) |
| Loss | so nos tokens da resposta |

Por que LoRA: treina so ~0,5% dos parametros, usa pouca memoria de GPU e salva um adaptador pequeno (alguns MB).

## 3. GPU: por que nao e CUDA
CUDA e exclusivo de placas NVIDIA. A **RX 6600 e AMD**, entao usamos o backend **DirectML** (`torch-directml`). O codigo (`gpu_utils.py`) tambem aceita CUDA, entao alguem com NVIDIA consegue rodar o mesmo projeto.
Se o computador tiver mais de uma GPU (ex.: grafico integrado), use `--gpu N` com o indice mostrado por `verificar_gpu.py`.

## 4. Como rodar
Requisitos: Python **3.10** (recomendado) e driver AMD atualizado.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python verificar_gpu.py        # confirma a GPU
python criar_dataset.py        # gera data/dataset.jsonl e data/perguntas_teste.json

python train.py                # treino unico com os parametros padrao
python experimentos.py         # treinos com varios parametros + grafico
python avaliar.py --adapter modelos/base_lr3e-4_r8_e5
python app.py --adapter modelos/base_lr3e-4_r8_e5     # frontend em http://127.0.0.1:7860
```
Na primeira execucao o modelo (~1 GB) e baixado do Hugging Face, entao e preciso internet.

## 5. Estrutura
```
verificar_gpu.py   informacoes e teste da GPU
gpu_utils.py       selecao do dispositivo (DirectML/CUDA)
llm_utils.py       carregar modelo e gerar texto
criar_dataset.py   dataset de treino e perguntas de teste
train.py           fine-tuning LoRA
experimentos.py    comparacao de parametros
avaliar.py         base x ajustado
app.py             frontend Gradio
data/              dataset e perguntas de teste
resultados/        metricas, graficos e respostas
modelos/           adaptadores LoRA treinados
docs/              prints (GPU no VSCode, frontend)
```

## 6. Parametros testados
`experimentos.py` treina as combinacoes abaixo e grava tudo em `resultados/experimentos.csv`.
Depois de rodar, **copie os valores reais para esta tabela**:

| Execucao | lr | rank | alpha | epocas | loss treino | loss validacao | tempo (s) |
|---|---|---|---|---|---|---|---|
| base_lr3e-4_r8_e5 | 3e-4 | 8 | 16 | 5 | | | |
| lr1e-4_r8_e5 | 1e-4 | 8 | 16 | 5 | | | |
| lr1e-3_r8_e5 | 1e-3 | 8 | 16 | 5 | | | |
| lr3e-4_r16_e5 | 3e-4 | 16 | 32 | 5 | | | |
| lr3e-4_r8_e10 | 3e-4 | 8 | 16 | 10 | | | |

Grafico: `resultados/comparacao_loss.png`.
Na geracao tambem se variam `temperature` e `top_p` (ver secao 7).

## 7. Avaliacao do modelo
`avaliar.py` faz as mesmas 15 perguntas (que **nao estao no treino**) ao modelo base e ao ajustado, em duas temperaturas (0 e 0,7):
- **cobertura de palavras-chave** (0 a 1), calculada automaticamente;
- **tokens por segundo** na GPU;

Limitacoes: o dataset e pequeno, a cobertura de palavras-chave e uma metrica simples e o modelo de 0,5 B pode errar em perguntas fora do que viu.

## 8. Frontend
`app.py` abre uma interface Gradio com campo de prompt, escolha entre modelo base e ajustado, sliders de temperatura, top_p, maximo de tokens e penalidade de repeticao, alem do desempenho em tokens/s.

## 9. Evidencias

GPU utilizada

<img width="1905" height="1027" alt="image" src="https://github.com/user-attachments/assets/ba21eab4-166f-4608-aa06-540e3e861bec" />


Modelo rodando com modelo da GPU, Código fonte e front-end exbidos

<img width="1915" height="1025" alt="image" src="https://github.com/user-attachments/assets/6f203432-9d23-437e-9001-4c6cedb77f05" />


