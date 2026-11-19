# Leitor de dígitos com YOLO

Esta etapa troca a leitura das sete sondas por um detector de caracteres no
recorte do visor. A perspectiva, o rastreamento local, a confirmação temporal,
o ID do teste, o TXT JSONL e a exportação CSV continuam integrados.

```text
Webcam → alinhamento da moldura → perspectiva → recorte colorido
       → YOLO: 0–9, ponto e sinal → ordenar e validar → TXT JSONL → CSV
```

O perfil inicial é `perfis/multimetro-yolo.json`, derivado de
`perfis/multimetro-perspectiva.json`. Seus pesos apontam para
`models/digits/best.pt`. Esse arquivo só existirá após o treinamento.
O programa recusa pesos genéricos COCO e não baixa modelos durante a leitura.

## 1. Ambiente

O ambiente `.venv-yolo` é separado do Python do sistema. Se já estiver instalado:

```bash
cd /home/senai/estagio_
source .venv-yolo/bin/activate
```

Para configurar em outro computador:

```bash
python3 -m venv .venv-yolo
source .venv-yolo/bin/activate
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-yolo.txt
```

Os comandos abaixo usam CPU. GPU pode ser selecionada com `--device 0` quando
o driver e a instalação de PyTorch suportarem CUDA. As versões de CPU são
suficientes para verificar o fluxo; o tempo de treinamento depende dos dados e
do hardware. O treinamento e a inferência usam a
[API oficial da Ultralytics](https://docs.ultralytics.com/modes/train/).

## 2. Anotar as primeiras imagens

Seis recortes reais anteriores foram importados na sessão `referencias-iniciais`.
Eles estão **sem rótulos** e não entram no treino até serem revisados:

```bash
python -m display_logger yolo-annotate \
  --dataset datasets/digits --session referencias-iniciais
```

Na janela:

- Aperte `0` a `9` para escolher a classe e arraste uma caixa em torno de cada dígito.
- Aperte `.` para o ponto decimal e `-` para o sinal negativo.
- Marque os traços visíveis do caractere inteiro. No dígito `1`, não inclua a
  largura vazia de uma célula completa. Não inclua o ponto na caixa do dígito.
- Marque **todos** os caracteres visíveis, incluindo zeros, ponto e sinal.
- `Backspace` desfaz a última caixa e `R` limpa a imagem atual.
- `B` marca visor apagado, `OL` ou outro estado sem medição como exemplo negativo.
  Confira a indicação `NEGATIVO` antes de salvar; esses exemplos têm rótulo vazio.
- `Enter` salva e avança. Confira no terminal o texto montado com seus rótulos.
- `N/P` navegam sem salvar a edição atual. `Q/ESC` encerra.

As imagens de referência incluem alguns valores próximos de zero. Elas são um
ponto de partida para aprender a anotação; não cobrem todos os caracteres ou as
condições necessárias ao treinamento.

## 3. Coletar dados variados

Encerre outras capturas para liberar a webcam. Use o perfil geométrico já calibrado:

```bash
python -m display_logger yolo-collect \
  --camera 0 --camera-backend v4l2 \
  --profile perfis/multimetro-perspectiva.json \
  --dataset datasets/digits --session bancada-01 \
  --samples 50 --interval 1 --duration 60
```

As imagens são capturadas automaticamente no intervalo informado. `Q` encerra.
`--no-preview` permite captura sem janela; `--video arquivo.avi` lê um vídeo.
O comando não usa o detector para criar rótulos: ele salva o recorte colorido.

Repita a coleta na mesma sessão para acrescentar exemplos de outros valores.
Inclua todos os dígitos, diferentes posições de ponto, sinal negativo, reflexos
moderados, variações de iluminação e exemplos negativos. Prepare sessões
independentes para validação e teste, por exemplo `validacao-01` e `teste-01`.
Não divida quadros quase idênticos do mesmo vídeo entre treino e teste.

Para importar fotos do quadro inteiro:

```bash
python -m display_logger yolo-import \
  --profile perfis/multimetro-perspectiva.json \
  --dataset datasets/digits --session fotos-01 \
  --images foto-1.png foto-2.png
```

Se as fotos já forem recortes frontais do visor, omita `--profile`.
Anote cada sessão com `yolo-annotate --session NOME`.

Para selecionar uma nova região sem marcar células individuais, a calibração
também aceita o motor YOLO antes de os pesos existirem:

```bash
python -m display_logger calibrate \
  --camera 0 --perspective --engine yolo \
  --profile perfis/outro-visor-yolo.json \
  --weights models/digits/best.pt --digits 4 --unit volts
```

Nesse modo você marca quatro cantos e a ROI numérica; não marca os segmentos.
Para coletar, use esse novo perfil em `yolo-collect`.

## 4. Conferir e preparar o dataset

```bash
python -m display_logger yolo-status --dataset datasets/digits

python -m display_logger yolo-prepare \
  --dataset datasets/digits --output datasets/digits/prepared \
  --val-session validacao-01 --test-session teste-01
```

São necessárias pelo menos três sessões revisadas. A sessão inteira pertence a
um único split. Imagens sem revisão são ignoradas; um rótulo vazio revisado é um
exemplo negativo, enquanto uma imagem sem arquivo de rótulo permanece pendente.
Imagens idênticas em splits diferentes são recusadas. Rótulos ou imagens alterados
após a revisão precisam ser revisados de novo.

Se omitir os dois argumentos de sessão, o programa escolhe uma sessão para
validação e outra para teste, de forma reproduzível, e deixa as demais no treino.
A pasta de saída deve ser nova. Para outra versão, use `prepared-v2`, por exemplo.

As classes são fixas:

| ID | Caractere |
| --- | --- |
| 0–9 | Dígitos 0–9 |
| 10 | `dot`: ponto decimal |
| 11 | `minus`: sinal negativo |

O resultado segue o
[formato YOLO de detecção](https://docs.ultralytics.com/datasets/detect/).
`data.yaml` é escrito como JSON, que também é um documento YAML válido.
`manifest.json` registra sessões, textos completos e hashes dos arquivos.
O treino exige exemplos de todas as doze classes no conjunto de treinamento.
Ter todas as classes é uma condição mínima; não comprova quantidade suficiente
nem boa cobertura de iluminação, fontes e posições.

## 5. Treinar

```bash
python -m display_logger yolo-train \
  --data datasets/digits/prepared/data.yaml \
  --model yolo26n.pt --epochs 100 --imgsz 640 --batch 8 \
  --device cpu --output models/digits/best.pt
```

O treinamento pode baixar os pesos-base informados em `--model`. Ele adapta as
classes ao nosso dataset. Espelhamentos horizontal e vertical ficam desativados,
pois podem alterar o significado dos dígitos. Um modelo de saída existente é
preservado; escolha outro `--output` para uma nova versão.

## 6. Avaliar a leitura completa

```bash
python -m display_logger yolo-evaluate \
  --data datasets/digits/prepared/data.yaml \
  --weights models/digits/best.pt --device cpu \
  --expected-digits 4 --output reports/yolo-evaluation.json
```

O relatório mede acerto do texto completo nas imagens de teste, erros aceitos,
rejeições de leituras válidas e falsas aceitações em exemplos negativos. Ele
inclui resultado por imagem. O ponto e o sinal fazem parte da comparação.
As métricas de detecção do treinamento, isoladamente, não provam que o número
completo foi lido corretamente.

## 7. Usar o modelo no ensaio

```bash
python -m display_logger yolo-profile \
  --from-profile perfis/multimetro-perspectiva.json \
  --profile perfis/multimetro-yolo.json \
  --weights models/digits/best.pt --expected-digits 4

python -m display_logger run \
  --camera 0 --profile perfis/multimetro-yolo.json \
  --test-id teste-yolo-001 --out logs/teste-yolo-001.txt \
  --debug-dir logs/teste-yolo-001-diagnostico

python -m display_logger csv \
  --input logs/teste-yolo-001.txt --output logs/teste-yolo-001.csv
```

O perfil gerado exige um ponto decimal por padrão. Para instrumentos que também
exibem inteiros, use `--allow-integer` em `yolo-profile` e em `yolo-evaluate`.
O número esperado de dígitos deve corresponder ao instrumento; zeros suprimidos
ou layouts variáveis precisam de outra configuração em `yolo_expected_digits`.
O motor mantém candidatos acima de `yolo_detection_conf` e rejeita a leitura se
a confiança de qualquer caractere ficar abaixo de `min_confidence`, evitando
descartar silenciosamente um ponto ou sinal de baixa confiança.

A prévia e `yolo-deteccoes.png` mostram as caixas detectadas. A ausência dos pesos
treinados, classes incompatíveis, caracteres faltantes e detecções conflitantes
geram erros ou leituras inválidas, sem recorrer automaticamente ao leitor antigo.

A unidade continua vindo do perfil. Esta etapa não identifica V/mV, AC/DC nem
localiza o visor globalmente; ela substitui o leitor de caracteres dentro da ROI.
