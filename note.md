# note — estado del trabajo

Fecha: 2026-11-23. `main` en `3593184`. Abierto: el primer PR de `train/`
(rama `train`). Este fichero es el parte de estado: lo hecho, lo que
quedó a medias al parar, y lo planeado. El plan de ejecución sigue en
`roadmap.md`; el contrato del tensor, en `tensor/DESIGN.md`.

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
| Spike M0: boundary FFI, gradiente a mano verificado contra diferencia central, libtorch CPU, 8 gaps medidos | `spike/m0/` | #8 |
| Spike ggml/Vulkan: ggml pineado `v1.0.0`, builds AVX2 y Vulkan, medidas iGPU vs CPU vs libtorch | `spike/ggml-vulkan/` | #14 |
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

### Fuera de este repo

- **agentica-lang/agentica#1918** (abierto): un `resource` es nominal a su
  `extern module`, así que un handle no puede vivir en datos ni cruzar
  paquetes. Es la causa de que exista (b′).
- **agentica-lang/agentica#1919** (mergeado): `docs/future/30-native-resources.md`
  con la regla de hoy, el caso de `ml`, la propuesta, los cinco invariantes
  y lo decidido en contra.

## 2. En progreso (parado a propósito)

- **Reglas de backward** (rama `grad-rules`, PR #24): `stop_grad` (op de
  wrapper, sin primitiva de backend) y reglas para `sqrt`, `sigmoid`, `tanh`,
  `gelu_tanh_approx`, `max` (empates a partes iguales), `squeeze`,
  `unsqueeze`, `slice`, `concat`, `stack` y `gather` (one-hot `matmul`; el
  scatter-add es ml#25). Con ellas `softmax`, `layer_norm`, `rms_norm`,
  atención, `embedding`, `positional` y `silu`/`gelu` son diferenciables, y
  `train/` gana `Loss.cross_entropy` (labels enteros, estable vía
  `stop_grad`).

Ningún subagente sigue corriendo.

## 3. Futuro planeado (orden propuesto)

1. **`grad`**: faltan reglas para `pow`, `clamp`, `batch_matmul`,
   `to_dtype` entre floats, `conv1d` y `avg_pool1d`, y el scatter-add de
   `gather` (ml#25).
2. **`nn/`**: falta decoder (máscara causal, cross-attention), conv,
   `rms_norm`, y `gelu` en libtorch (`gelu_tanh_approx`, en `tensor/`).
3. **`train/`**: mergeado (#21); softmax cross-entropy en #24. Después:
   shuffling, schedules, y aplanar structs de capas (pytrees)
   cuando un modelo lo pida.
4. **`jit`/caché del grafo**: `fit` retraza cada paso (~75k pasos de VM por
   época del XOR en la referencia).
5. **`hub/`**: safetensors + descarga HF. Descargas con `http-client`
   `stream()` (redirects y retry incluidos), **nunca** `bond` (es solo
   servidor) ni `std/http download()` (tope 256 MiB, sin redirects).
   Resume/progreso/sha256 hay que construirlos.
6. **Batching de grafo / JIT**: justificado por los 2 ms por dispatch. Es
   el camino para que la iGPU sirva, y el mismo mecanismo que `xla/`
   (PJRT) necesitaría.
7. **Backend Vulkan/iGPU** como backend real, ya con batching; requiere
   añadir una variante a `Device` (hoy `Cpu | Cuda(Int)`).
8. **`candle/`** (eager, Rust) y/o **ggml** como backend general.
9. **dGPU CUDA**: build CUDA de libtorch, `Device.Cuda(Int)`, `to_device`,
   y un job de CI no requerido (no se puede bajar en cada push).
10. Cuando cierre **#1918**: evaluar volver a la opción **A** (tabla de
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
