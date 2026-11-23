# note — estado del trabajo

Fecha: 2026-11-23. `main` en `c9e23fb` (decoder de `nn/` mergeado, #26).
Abierto: Whisper vía B, primer PR (`asr/`, rama `asr`). Este fichero es el
parte de estado: lo hecho, lo que quedó a medias al parar, y lo planeado. El
plan de ejecución sigue en `roadmap.md`; el contrato del tensor, en
`tensor/DESIGN.md`.

## 1. Hecho (mergeado en `main`)

| Qué | Dónde | PRs |
|---|---|---|
| Paquete `whisper/`: binding real sobre whisper.cpp (submódulo `v1.9.5` pineado), WAV 16 kHz en Agentica puro, CI matricial por paquete | `whisper/` | #2, #4, #7, #9 |
| `whisper/`: idioma (`es`/`auto`), `translate`, segmentos con timestamps en ms, `transcribe` = join puro | `whisper/whisper.agc` | #7 |
| `whisper/`: micrófono por `Process` (`arecord`) + audio largo con VAD, ventanas con solape y dedupe | `whisper/mic.agc`, `stream.agc` | #9 |
| `tensor/` M1: API en la raíz + `src/`, backend de referencia puro, guard de no finitos, 148 tests | `tensor/` | #10 |
| `tensor/` M2 = **decisión (b′)**: el backend libtorch vive **dentro** de `tensor/` (`src/torch.agc`, `src/native.agc`, `native/torch.c`, `[native.torch] optional`), `Handle = Native(TensorHandle) \| Plain(Data)`, dispatch cableado en los 47 brazos | `tensor/` | #11 |
| `tensor/`: cobertura **42/45 en libtorch, 45/45 en referencia**, con `div`, `conv1d`, `avg_pool1d` reales; `softmax`/`layer_norm`/atención ya corren en torch | `tensor/` | #13 |
| `tensor/`: tracing y gradientes (§10): `Trace` como quinto brazo, `value_and_grad`, reglas de backward de las primeras ops, suite de paridad traced vs eager | `tensor/trace.agc`, `src/graph.agc` | #16, #18 |
| `tensor/`: tests property-based con `prop` (órdenes de `axes`, `concat`, round trips de dtype, oráculo torch, conv/pool contra bucles) | `tensor/test/prop_*` | #17, #19 |
| `nn/` primer PR: capas funcionales (`dense`, `layer_norm`, `embedding`, `attention`, `mlp`, `encoder_block`), init con semilla, solo forward; `dense` y `mlp` relu diferenciables vía `Trace` | `nn/` | #20 |
| `train/` primer PR: `Loss.mse`, `Loss.binary_cross_entropy`, `Optim` (SGD, Adam) puros, `Train.fit` sobre `Trace.value_and_grad`; la raíz pasa a ser un workspace | `train/`, `agentica.toml` | #21 |
| `tensor/`: `stop_grad` y reglas de backward de `sqrt`, `sigmoid`, `tanh`, `gelu_tanh_approx`, `max`, `squeeze`, `unsqueeze`, `slice`, `concat`, `stack`, `gather`; `softmax`, `layer_norm`, atención y `embedding` diferenciables; `Loss.cross_entropy` | `tensor/`, `train/` | #24 |
| `nn/` decoder: `causal_mask`, `DecoderBlock` pre-norm como Whisper; `Load.decoder_block` en `hub/` | `nn/`, `hub/` | #26 |
| Spike M0: boundary FFI, gradiente a mano verificado contra diferencia central, libtorch CPU, 8 gaps medidos | `spike/m0/` | #8 |
| Spike ggml/Vulkan: ggml pineado `v1.0.0`, builds AVX2 y Vulkan, medidas iGPU vs CPU vs libtorch | `spike/ggml-vulkan/` | #14 |
| Spike ggml-graph: un paso de entrenamiento trazado (forward, backward, SGD/Adam) bajado a **un** grafo ggml, construido una vez, en la iGPU por Vulkan | `spike/ggml-graph/` | #30 |
| Docs: `tensor/DESIGN.md` (contrato, sin fases), `roadmap.md` (orden e hitos), `README.md`, `.gitattributes`, CI matricial | raíz | #3, #6 |

### Medidas que importan (esta máquina: Ryzen 5 3550H, Vega iGPU RADV, AVX2+FMA, **sin AVX512**)

- **2.00 ms por grafo** despachado a Vulkan. Por eso la iGPU **solo gana
  con ≥~1 GFLOP** (512×2048: 3.1×, 1024³: 1.8× frente a 8 hilos AVX2) y
  **pierde por debajo** (256³: 2.9× más lenta).
- `mul_mat` de ggml CPU es 1.5–3.4× más lento que el BLAS de libtorch.
- Consecuencia de diseño: **Vulkan no puede ser un backend op-por-op**;
  pide batching/compilación de grafo. Es el primer dato objetivo a favor
  del backend JIT.
- AVX512 no es alcanzable aquí: el techo real es AVX2.
- **Un paso compilado a un grafo ggml sí sirve en la iGPU** (spike
  ggml-graph, MLP 784→512→512→10, SGD): 4.7 / 5.6–6.7 / 14–16 / 45–51 ms
  por paso a batch 64 / 256 / 1024 / 4096, contra 3.1–3.9 / 11 / 41–48 /
  144–170 de libtorch C++ en CPU: gana **1.6–3.7× desde batch 256**, pierde
  a 64. Ninguna op del paso falta en Vulkan. Trazar, derivar y bajar cuesta
  2–6 ms una vez; luego un paso es una llamada FFI.
- `Train.fit` sobre libtorch hoy: 42–105 ms por paso (op a op, y el
  optimizador materializa cada `full` por FFI); a batch 4096 se para: el
  `full` de torch usa `List.repeat`, que tiene tope de 1 048 576.

### Fuera de este repo

- **agentica-lang/agentica#1918** (abierto): un `resource` es nominal a su
  `extern module`, así que un handle no puede vivir en datos ni cruzar
  paquetes. Es la causa de que exista (b′).
- **agentica-lang/agentica#1919** (mergeado): `docs/future/30-native-resources.md`
  con la regla de hoy, el caso de `ml`, la propuesta, los cinco invariantes
  y lo decidido en contra.

## 2. En progreso (parado a propósito)

- **Whisper vía B, primer PR** (rama `asr`): el miembro `asr/` con el
  forward de Whisper (stem conv, encoder, decoder, logits atados) cargado
  con los nombres de HF y el `config.json`; `Nn.Conv1d` y `Load.conv1d`.
  Solo referencia: en libtorch falta `gelu_tanh_approx`. La deriva de la
  GELU tanh frente a la erf, medida en el modelo de test: 1.7e-4 (encoder),
  3.8e-5 (logits).

Ningún subagente sigue corriendo.

## 3. Futuro planeado (orden propuesto)

1. **`grad`**: faltan reglas para `pow`, `clamp`, `batch_matmul`,
   `to_dtype` entre floats, `conv1d` y `avg_pool1d`, y el scatter-add de
   `gather` (ml#25).
2. **`nn/`**: decoder mergeado (#26), conv en `asr`. Siguiente: caché KV
   (con ella, un `mask` opcional en `decoder_block` para padding), luego
   `rms_norm`, y `gelu` en libtorch (`gelu_tanh_approx`, en `tensor/`).
3. **`asr/`** (Whisper vía B), después del primer PR: frontend log-mel,
   tokenizer, decodificación greedy y beam (sobre la caché KV), y la descarga
   de pesos reales (con `hub/`). Fuera del primer PR por YAGNI. Lo
   siguiente: comparar con whisper-tiny real (hoy solo se compara con el peer
   propio).
4. **`train/`**: mergeado (#21); softmax cross-entropy mergeado (#24). Después:
   shuffling, schedules, y aplanar structs de capas (pytrees)
   cuando un modelo lo pida.
5. **`jit`/caché del grafo**: `fit` retraza cada paso (~75k pasos de VM por
   época del XOR en la referencia). El spike ggml-graph mide retrazar el
   paso de un MLP en 1–3 ms: basta una caché por estructura del grafo
   (ops, formas; no valores de `full` ni de constantes), sin API `jit`.
6. **`hub/`**: primer PR mergeado (#23). Falta la descarga HF, con `http-client`
   `stream()` (redirects y retry incluidos), **nunca** `bond` (es solo
   servidor) ni `std/http download()` (tope 256 MiB, sin redirects).
   Resume/progreso/sha256 hay que construirlos.
7. **Batching de grafo / JIT**: justificado por los 2 ms por dispatch y
   **probado** por el spike ggml-graph (#30): el grafo que `Trace` ya
   graba se baja entero a ggml. Es el mismo mecanismo que `xla/` (PJRT)
   necesitaría.
8. **Backend Vulkan/iGPU** como backend real (diseño y PRs en
   `spike/ggml-graph/README.md`): `Device.Vulkan(Int)`, `Backend.Ggml(Device)`
   y `Handle.Ggml` en `tensor/` como (b′) hasta #1918; `Trace.evaluate` sobre
   `Ggml` compila el grafo entero con caché por estructura; `Train.fit` mete
   el optimizador en el mismo grafo y deja parámetros y momentos en el
   backend. Decidir fp16 (`mul_mat` de Vulkan, 1.3× a 4096) o fp32 con
   tolerancia por dispositivo.
9. **`candle/`** (eager, Rust) y/o **ggml** como backend general.
10. **dGPU CUDA**: build CUDA de libtorch, `Device.Cuda(Int)`, `to_device`,
   y un job de CI no requerido (no se puede bajar en cada push).
11. Cuando cierre **#1918**: evaluar volver a la opción **A** (tabla de
    `Fn` por backend), que deja `tensor/` sin nativo. Con un solo backend
    no compensa; con dos o más, sí.

## 4. Deuda y issues abiertos

- **agentica-lang/ml#1**: diseño original (Q1–Q4). Q1 quedó resuelta de
  facto por (b′); Q2 (operadores) sigue en "no"; Q3 (lifetime) y Q4
  (checking) siguen abiertas.
- **ml#5**: menores de `whisper/` (8 de #4 y 3 de #2, más 2 de #7).
- **ml#12**: menores de `tensor/` (comentarios obsoletos, `catch` de OOM,
  dos aserciones, y el límite de CI que no compila el C++).
- **agentica#1918**: el cambio de lenguaje.
- `Torch.full` (`tensor/src/torch.agc`) arma los valores con `List.repeat`
  y los sube: por encima de 1 048 576 elementos la ejecución se para
  (`RUNTIME-INVALID-COUNT`, no un `Result`), y cada `full` del optimizador
  cruza el FFI cada paso (spike ggml-graph).
- `conv1d`/`avg_pool1d` no soportan dilation ni groups (declarado).
- `gelu_tanh_approx`, `batch_matmul` y `to_device` no están en libtorch
  (declarado en el README del paquete, con razón).
- La copia **instalada** de `whisper/` no compila: el header del motor está
  en el submódulo. Falta la tabla `prebuilt` por plataforma.

## 5. Límites conocidos (no son bugs, son el trato)

- **CI no compila el C++ de libtorch**: no se puede bajar un binario de
  473 MB por push. Un job verde prueba `check`, `fmt`, el header
  sincronizado y el camino "módulo ausente"; un error de compilación o una
  deriva de la API aparece como un warning en un job verde. La señal es
  `warning: native module torch not built`.
- **El runner solo imprime PASS**: un test que hace SKIP no deja rastro
  visible, así que los skips deben ser estructurales (aserción del estado
  ausente), no un `puts` y `ok`.
- **Vulkan no está en CI** (no hay driver pineable): el spike es local.

## 6. Cómo reanudar

- Un lane = una rama y un worktree hermano: `git worktree add
  ../agentica-ml-<tema> -b <tema> origin/main`. Un solo escritor por
  worktree.
- Se pasa modelo explícito: ejecutores `opencode-go/deepseek-v4.1-flash`,
  revisores `opencode-go/glm-5.3-flash`. **Omitir `model` NO hereda el de la
  sesión**: resuelve al `defaultModel` de `~/.pi/agent/settings.json`.
- Loop por PR: rama → PR → review de un agente fresco → un batch de fixes
  (críticos y mayores, más menores baratos) → CI verde → review otra vez →
  merge **solo con el ok del dueño**. Menores no bloquean: van a un issue.
- `git push` por https da 403 con el PAT; empujar con
  `git push git@github.com:agentica-lang/ml.git <rama>`.
- Cada paquete valida con `agentica check`, `agentica fmt --check`, el
  `--check` del header de native-gen, y `agentica test`
  (`agentica install` antes si hay dependencias).
- Antes de tocar `tensor/`, leer `tensor/DESIGN.md`. Cambiar el contrato
  exige actualizarlo en el mismo PR; el roadmap no entra en el DESIGN.
