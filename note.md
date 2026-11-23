# note — estado del trabajo

Fecha: 2026-11-23. `main` en `33fc239` (loader de `train/` mergeado, #28).
Este fichero es el parte de estado: lo hecho, lo que sigue abierto y lo
planeado. El plan de ejecución sigue en `roadmap.md`; el contrato del tensor,
en `tensor/DESIGN.md`.

## 1. Hecho (mergeado en `main`)

| Qué | Dónde | PRs |
|---|---|---|
| Paquete `whisper/`: binding real sobre whisper.cpp (submódulo `v1.9.5` pineado), WAV 16 kHz en Agentica puro, CI matricial por paquete | `whisper/` | #2, #4, #7, #9 |
| `whisper/`: idioma (`es`/`auto`), `translate`, segmentos con timestamps en ms, `transcribe` = join puro | `whisper/whisper.agc` | #7 |
| `whisper/`: micrófono por `Process` (`arecord`) + audio largo con VAD, ventanas con solape y dedupe | `whisper/mic.agc`, `stream.agc` | #9 |
| `tensor/` M1: API en la raíz + `src/`, backend de referencia puro, guard de no finitos, 148 tests | `tensor/` | #10 |
| `tensor/` M2 = **decisión (b′)**: el backend libtorch vive **dentro** de `tensor/` (`src/torch.agc`, `src/native.agc`, `native/torch.c`, `[native.torch] optional`), `Handle = Native(TensorHandle) \| Plain(Data)`, dispatch cableado en cada brazo | `tensor/` | #11 |
| `tensor/`: cobertura **42/45 en libtorch, 45/45 en referencia**, con `div`, `conv1d`, `avg_pool1d` reales; `softmax`/`layer_norm`/atención ya corren en torch | `tensor/` | #13 |
| `tensor/`: tracing y gradientes (§10): `Trace` como quinto brazo, `value_and_grad`, reglas de backward de las primeras ops, suite de paridad traced vs eager | `tensor/trace.agc`, `src/graph.agc` | #16, #18 |
| `tensor/`: tests property-based con `prop` (órdenes de `axes`, `concat`, round trips de dtype, oráculo torch, conv/pool contra bucles) | `tensor/test/prop_*` | #17, #19 |
| `nn/` primer PR: capas funcionales (`dense`, `layer_norm`, `embedding`, `attention`, `mlp`, `encoder_block`), init con semilla, solo forward; `dense` y `mlp` relu diferenciables vía `Trace` | `nn/` | #20 |
| `train/` primer PR: `Loss.mse`, `Loss.binary_cross_entropy`, `Optim` (SGD, Adam) puros, `Train.fit` sobre `Trace.value_and_grad`; la raíz pasa a ser un workspace | `train/`, `agentica.toml` | #21 |
| `tensor/`: `stop_grad` y reglas de backward de `sqrt`, `sigmoid`, `tanh`, `gelu_tanh_approx`, `max`, `squeeze`, `unsqueeze`, `slice`, `concat`, `stack`, `gather`; `softmax`, `layer_norm`, atención y `embedding` diferenciables; `Loss.cross_entropy` | `tensor/`, `train/` | #24 |
| `train/` loader: `Data` (dataset en memoria o de un safetensors, barajado con semilla por época, `drop_last`, un `gather` por batch), prefetch en un `Task` con `Channel` acotado; `fit` toma un `Data.Loader` | `train/data.agc` | (este PR) |
| `nn/` decoder: `causal_mask`, `DecoderBlock` pre-norm como Whisper; `Load.decoder_block` en `hub/` | `nn/`, `hub/` | #26 |
| `asr/` (Whisper vía B): forward del modelo (stem conv, encoder, decoder, logits atados) cargado con los nombres de HF y el `config.json`; `Nn.Conv1d` y `Load.conv1d`. Solo referencia: en libtorch falta `gelu_tanh_approx`; deriva de la GELU tanh frente a la erf en el modelo de test: 1.7e-4 (encoder), 3.8e-5 (logits) | `asr/`, `nn/`, `hub/` | #27 |
| Todos los miembros en agentica 0.36.0; `agentica test --max-steps` por test en CI, sin los apaños del presupuesto de pasos; pins entre miembros al día (instalan desde fuera del workspace); checks de gradiente completos en `nn/` | `*/AGENTICA_VERSION`, `.github/workflows/ci.yml` | (este PR) |
| `tensor/`: fuera los stubs `Backend.Xla`/`Candle` y `Device.Cuda` (sin módulo, solo devolvían `error`); `Device` queda con el brazo `Cpu` | `tensor/` | (este PR) |
| Spike M0: boundary FFI, gradiente a mano verificado contra diferencia central, libtorch CPU, 8 gaps medidos | `spike/m0/` | #8 |
| Spike ggml/Vulkan: ggml pineado `v1.0.0`, builds AVX2 y Vulkan, medidas iGPU vs CPU vs libtorch | `spike/ggml-vulkan/` | #14 |
| Docs: `tensor/DESIGN.md` (contrato, sin fases), `roadmap.md` (orden e hitos), `README.md`, `.gitattributes`, CI matricial | raíz | #3, #6 |

### Medidas que importan (esta máquina: Ryzen 5 3550H, Vega iGPU RADV, AVX2+FMA, **sin AVX512**)

- **2.00 ms por grafo** despachado a Vulkan. Por eso la iGPU **solo gana
  con ≥~1 GFLOP** (512×2048: 3.1×, 1024³: 1.8× frente a 8 hilos AVX2) y
  **pierde por debajo** (256³: 2.9× más lenta).
- `mul_mat` de ggml CPU es 1.5–3.4× más lento que el BLAS de libtorch.
- Consecuencia de diseño: **Vulkan no puede ser un backend op-por-op**;
  pide batching de grafo.
- AVX512 no es alcanzable aquí: el techo real es AVX2.

### Fuera de este repo

- **agentica-lang/agentica#1918** (abierto): un `resource` es nominal a su
  `extern module`, así que un handle no puede vivir en datos ni cruzar
  paquetes. Es la causa de que exista (b′).
- **agentica-lang/agentica#1919** (mergeado): `docs/future/30-native-resources.md`
  con la regla de hoy, el caso de `ml`, la propuesta, los cinco invariantes
  y lo decidido en contra.

## 2. En progreso

Nada abierto en `main`. Fuera de él hay trabajo en ramas de spike.

## 3. Siguientes pasos

1. **`grad`**: faltan reglas para `pow`, `clamp`, `batch_matmul`,
   `to_dtype` entre floats, `conv1d` y `avg_pool1d`, y el scatter-add de
   `gather` (ml#25).
2. **`nn/`**: caché KV (con ella, un `mask` opcional en `decoder_block`), y
   `gelu` en libtorch (`gelu_tanh_approx`, en `tensor/`).
3. **`asr/`**: frontend log-mel, tokenizer y decodificación; después, comparar
   con whisper-tiny real (hoy solo se compara con el peer propio).
4. **`train/`**: schedules, y aplanar structs de capas (pytrees) cuando un
   modelo lo pida. El prefetch solo oculta el coste de armar el batch; un
   tensor nativo muere con el task que lo hizo.
5. **`hub/`**: descarga HF con `http-client` `stream()`, **nunca** `bond` (es
   solo servidor) ni `std/http download()` (tope 256 MiB, sin redirects).
6. **CI**: un job no requerido con libtorch pineado por sha256.
7. Cuando cierre **#1918**: mover el backend libtorch a un paquete `torch/`.

## 4. Deuda y issues abiertos

- **agentica-lang/ml#1**: diseño original (Q1–Q4). Q1 quedó resuelta de
  facto por (b′); Q2 (operadores) sigue en "no"; Q3 (lifetime) y Q4
  (checking) siguen abiertas.
- **ml#5**: menores de `whisper/` (8 de #4 y 3 de #2, más 2 de #7).
- **ml#12**: menores de `tensor/` (comentarios obsoletos, `catch` de OOM,
  dos aserciones, y el límite de CI que no compila el C++).
- **agentica#1918**: el cambio de lenguaje.
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
