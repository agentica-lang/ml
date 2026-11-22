# roadmap — tensor

> Visión: **tensores funcionales en Agentica (estilo Nx/Axon + JAX).**
> API funcional e inmutable (un training step devuelve parámetros nuevos),
> backends intercambiables, `grad` calculado **en Agentica sobre un grafo
> trazado** (como hace Nx), de modo que cada backend solo implementa forward.
> Diseño base en [ml#1](https://github.com/agentica-lang/ml/issues/1) y
> `README.md` de este repo. Este archivo es el plan de ejecución.

Fecha: 2026-11-22. Estado: **en marcha** (Vía A scaffold mergeada; no hay
bloqueos externos: el trabajo avanza por lanes paralelos).

Objetivo tractor (2026-11-22): **ejecutar Whisper** (small/medium, español,
entrada fichero + micrófono). Dos vías en paralelo: **(A) binding fino a
whisper.cpp** para transcribir en días y validar native-gen con algo real,
**(B) Whisper puro sobre `tensor`/`nn`/`hub`** como prueba end-to-end del
stack. La vía A no sustituye a la B: es el bootstrap mientras madura el
tensor puro. Checkout local de referencia: `../whisper.cpp` (con ggml
vendored, modelos de test y scripts `convert-pt-to-ggml.py`); en este repo
entra como submódulo en `third_party/`, no copiado (ver punto 7).

## 1. Qué copiamos de Nx (y qué no)

| Nx / ecosistema Elixir | Equivalente aquí |
|---|---|
| `Nx` = tensores + `Nx.Backend` behaviour (cada backend implementa los callbacks, con shape de salida como primer arg) | `tensor/` = API + interfaz de backend pequeña (~50–80 ops). Cada backend implementa forward; el dispatch es por valor, no por config global |
| `Nx.BinaryBackend` (binarios en la VM, el backend lento que todo lo verifica) | `tensor/` incluye **backend de referencia 100% Agentica**: lento, es el **oráculo de los tests** de cada backend |
| `Nx.Defn` (`defn`, `grad/2`, `value_and_grad`, JIT a CPU/GPU según compilador) | funciones Agentica que se **trazan a un grafo de ops**; `grad` se calcula en Agentica sobre ese grafo; el backend JIT (XLA/PJRT) compila el grafo entero |
| `Axon` (capas, inicializadores, loop de entrenamiento estilo Ignite, `Axon.Loop`, `Axon.Display`) | `nn/` (forward/inferencia) + `train/` (grad, losses, optimizadores, loop). Sin macros `defn`: Agentica es estático y ya chequea antes de correr |
| `Bumblebee` (modelos pre-entrenados + integración Hugging Face) | `hub/` (después): `safetensors` + modelos HF. Formato: 8 bytes LE con N, header JSON UTF-8 (`dtype`, `shape`, `data_offsets`, más `__metadata__` opcional), buffer contiguo little-endian C-order, sin strides, mmap-friendly, sin ejecución de código |
| `EXLA` (compilación JIT vía XLA/PJRT) | `xla/` vía **PJRT C API** (`pjrt_c_api.h`, sin capa propia): compila el grafo entero. Diferido (fase 6), es lo más complejo |

Lo que **no** copiamos: autograd de cada backend (XLA no tiene; candle tiene
el suyo propio). Por eso la decisión clave de ml#1 se mantiene: **grad en
Agentica, backends solo forward**.

## 2. Restricciones de Agentica que mandan en el diseño

1. **Sin sobrecarga de operadores sobre structs de usuario** (pregunta
   abierta nº 2 de ml#1, sigue abierta). Plan A: `Tensor.add(a, b)` + pipes
   (`t |> Tensor.add(other) |> Tensor.relu()`), no `a + b`. Si el lenguaje
   lo permite algún día, será azúcar sobre lo mismo.
2. **FFI = `extern module` + native-gen.** El `.agc` es la única fuente de
   las firmas. Flujo (`native-gen 0.4.0`, Agentica ≥ 0.35.0):
   `cli.agc new demo` → escribir C/Rust con tipos C → regenerar header
   (`cli.agc pkg.agc --output native/pkg_native.h`) → `--check` en CI para
   que el header nunca quede stale. Para Rust: `cli.agc build` escribe
   `rust/src/NAME_native.rs`, corre `cargo build --release`, copia la
   librería a `native/` y añade la entrada `[native.NAME]` por sistema.
3. **`def` vs `compute def`.** Un `def` no debe bloquear ni llamar a stacks
   propios (corutinas); cómputo largo o que espera → `compute def`
   (corre en el pool de cómputo). Los kernels push datos o devuelven un
   **resource**, y el loop vive en Agentica (el módulo nativo nunca llama
   de vuelta a Agentica).
4. **Handles = resources.** Un tensor vivo en libtorch/candle/ggml es un
   resource (`struct NAME_state *` + `NAME_FREE` thread-safe, solo datos C,
   nunca terms). Puntero válido solo durante la llamada: no guardarlo.
5. **Liberar intermedios a tiempo** (pregunta abierta nº 3). Miles de
   tensores por step no pueden esperar al GC. Opciones: buffers propiedad
   del grafo, o scopes explícitos. Depende de
   [agentica#1091](https://github.com/agentica-lang/agentica/issues/1091)
   (arenas nativas, 0.25.0) — verificar estado antes del diseño detallado.
6. **Shapes y dtypes se chequean en trace** (pregunta abierta nº 4). Cuánto
   carga el type-checker vs. runtime, por decidir. Cursores `List`/`Map` de
   native-gen se leen por `_next` (nada se copia a arrays); builders para
   construir resultados.
7. **Nada de copiar código externo: todo como submódulo git** (decisión 2026-11-22). whisper.cpp, ggml, libtorch, modelos y deps pesadas entran pineados en `third_party/` vía `git submodule`, nunca vendorizados. En este repo solo vive: `*.agc`, `agentica.toml`, el C/Rust fino propio, headers generados, tests y fixtures sintéticas.
8. **Distribución = `[native.NAME]` + prebuilt por plataforma.**
   `build = { sources = [...] }` para C; librerías comprometidas en
   `native/` por sistema para Rust/C++ pesado (libtorch: **CPU primero**,
   binarios grandes — irrelevante para `check`, crítico para CI/install).
   Reutilizar el workflow `native-prebuilt` de `agentica-lang/packages`.
8. **Versionado del repo**: un folder por paquete con su `agentica.toml`,
   tags con prefijo (`tensor-0.1.0`, `torch-0.1.0`, …), un root
   `[workspace]` cuando exista el primer paquete, `AGENTICA_VERSION` fijo
   por paquete y CI que corre `install / check / fmt-check / test /
   doc-examples` por paquete (patrón `packages/`).

## 3. Backends: decisión

| Backend | Kind | FFI | Prioridad |
|---|---|---|---|
| `reference` | eager, solo Agentica | ninguna | **fase 1** — oráculo |
| `torch` (libtorch) | eager | **capa C fina sobre la API C++** (como hace `tch-rs`): `torch::zeros/ones/empty/full/rand`, `Tensor.sizes()`, matmul, etc. CPU primero. Vive dentro de `tensor/` mientras #1918 no cierre | **fase 2** — el más completo, primer backend real; 18/45 primitivas hechas |
| `candle` | eager | capa C fina sobre Rust (`extern "C"`); bueno para inferencia, ligero | **fase 7** — después de entrenar sobre torch |
| `xla` | JIT estilo JAX | **PJRT C API directa**, sin capa (`dlopen` del plugin, cliente → dispositivos → compilación del grafo) | **fase 6** — lo más complejo, diferido |
| `ggml` | eager/inferencia LLM | C/C++ sin dependencias, CPU/GPU/NPU, quants 2–8 bits, ficheros `gguf`; el motor tras llama.cpp y whisper.cpp. **Cero allocs durante la ejecución**, sin fusión de kernels: op-set para transformers | **vía A (ya) + fase 8 (tensor puro)** — ver Fase W |
| `whisper` (binding) | inferencia STT | capa fina sobre `../whisper.cpp` (C++): `compute def transcribe`, modelo `gguf`/ggml small/medium, 16 kHz | **vía A** — primer valor funcional, no sustituye al tensor puro |

Candle sigue "tal vez" (cubre el mismo interfaz eager que torch, va
después). ggml se adelanta **solo como motor de la vía A** (whisper.cpp ya
lo trae vendored); el backend ggml puro sobre `tensor/` sigue en fase 8.

## 4. Fases

### Fase 0 — spike CPU (primera tarea del tensor puro)
Objetivo: encontrar los gaps de FFI **antes** de fijar la API.
Crear tensores, `matmul`, `grad` de una función pequeña, un paso SGD —
sobre `reference` y sobre libtorch. Solo CPU.
Sale: lista de gaps (tipos que cruzan, lifetimes, arenas) + la API mínima
congelada del backend.

### Fase 1 — `tensor/` + backend de referencia
- Tipos: `Tensor { shape, dtype, backend, handle }`, dtypes
  (`F32/F16/F64`, `I32/I64`, `U8`, `Bool` como mínimo; el resto según
  safetensors: `F16/F32/F64`, `I8/I16/I32/I64`, `U8`, `Bool`).
- Interfaz de backend pequeña (~50–80 ops): creación, elementwise,
  reducciones, `matmul`, reshape/slice, conv, random (semilla vía
  `System`/`Random`).
- Trazado a grafo + `grad`/`value_and_grad` en Agentica.
- Backend de referencia puro Agentica + suite de conformidad que **todos**
  los backends deberán pasar.
- Preguntas 1–4 de ml#1 cerradas aquí (struct de fns vs enum+`case`,
  operadores vs `Tensor.*`, liberación de intermedios, shapes/dtypes).

### Fase 2 — libtorch eager (dentro de `tensor/`), vía `native-gen`
- **Hecho**: `extern module "torch"` (`tensor/src/native.agc`, 13
  `compute def` + `resource TensorHandle`) con header generado por native-gen
  y `--check` en CI; capa C++ fina en `tensor/native/torch.c`; dispatch real
  en el brazo `Backend.Torch(device)` de `tensor.agc`; `Handle` con
  `Native(TensorHandle) | Plain(Data)`; 18 de las 45 primitivas §3.1 sobre
  libtorch, el resto `error("<op> is not implemented on the torch backend
  yet")`; suite de conformidad (`tensor/test/torch_*_test.agc`) que corre la
  misma llamada pública en `Reference` y `Torch` y compara
  valores/shape/dtype/errores; arenas por llamada y por step; libtorch como
  dependencia de sistema `optional` (nunca vendorizada ni descargada por el
  repo) documentada con su coste medido.
- **Por qué vive en `tensor/` y no en un paquete `torch/`**: un `resource` es
  nominal al bloque `extern module` que lo declara, así que el handle que
  lleva `Tensor` y el brazo que despacha tienen que estar en el mismo paquete
  (ver `tensor/DESIGN.md` §1, §2, §7). Es un trade-off deliberado, no el
  destino: [agentica-lang/agentica#1918](https://github.com/agentica-lang/agentica/issues/1918)
  (resources opacos que viven en datos y cruzan módulos) es la salida.
- **Pendiente cuando #1918 cierre**: mover `src/native.agc`, `src/torch.agc`,
  `native/torch.c` y su tabla `[native.torch]` a un paquete `torch/` propio
  (con su `AGENTICA_VERSION` y su tag `torch-0.1.0`), dejar `Handle` en
  `Native(Resource)` opaco y `tensor/` sin dependencia nativa, y devolver el
  job de CI a un `torch` que sólo necesite libtorch para la conformidad.
- **Pendiente en general** (fase 2 inacabada por primitivas, no por dispatch):
  las 27 primitivas §3.1 que faltan en el backend, con `div` y `tanh` primero
  porque desbloquean `softmax`/`gelu`/`silu` y la atención §3.2 en libtorch;
  `conv1d`/`avg_pool1d` cuando un consumidor las pida (§5); y el job de CI no
  requerido con libtorch pineado por sha256 que pruebe que el C++ compila y
  que los valores siguen coincidiendo (`tensor/README.md`, "CI").

### Fase 3 — `nn/` (solo forward = inferencia)
- Capas funcionales (dense, conv, norm, dropout, embeddings),
  activaciones, inicializadores. Parámetros = valores inmutables
  (mapas/structs); `apply(model, params, input)`.
- Solo depende de `tensor/`: funciona con reference y con torch.

### Fase 4 — `train/` (grad, losses, optimizadores, loop)
- `grad` sobre el grafo (fase 1) aplicado a `nn/`: losses, SGD/Adam,
  loop de entrenamiento estilo `Axon.Loop`, logging con `std/logger`.
- Criterio de salida: entrenar un MLP pequeño end-to-end sobre torch
  con la referencia como oráculo.

### Fase 5 — `hub/` (safetensors + Hugging Face, después)
- Parser safetensors en Agentica (header JSON + offsets; mmap donde el
  backend lo soporte) y cargador de pesos a `nn/`.
- Descarga de modelos (HF hub) y sharding multi-fichero con índice JSON.
- No hay ejecución de código al cargar (propiedad del formato).

### Fase 6 — `xla/` (PJRT JIT, diferido)
- Compilar el grafo entero vía PJRT C API; requiere que el grafo de
  fase 1 sea "compilable" (shapes estáticos). Es el equivalente a EXLA.

### Fase 7 — `candle/` (eager ligero)
- Mismo interfaz que `torch/` pero sobre la crate `candle-core`
  (`Device::Cpu/Cuda`, `Tensor`, `VarBuilder::from_mmaped_safetensors`).
  Un solo módulo Rust por paquete (`rust/` en raíz, `build` de native-gen).

### Fase 8 — `ggml/` (backend puro, futuro)
- Solo inferencia transformers cuantizados; depende de lo aprendido en
  `hub/` + `xla/`. La vía A (§4bis) ya usa ggml vía whisper.cpp sin
  esperar a esta fase. No entra en el MVP del tensor puro.

### Fase W — Whisper como objetivo tractor (transversal, dos vías)

**Vía A — `whisper/` binding a whisper.cpp (días, primer valor).**
- `extern module "whisper"` vía native-gen: `load_model(path) -> Resource`,
  `compute def transcribe(handle, pcm_16khz: List[Float]) -> Result[Text]`
  (+ segmentos con timestamps después). El loop y el resource viven en
  Agentica; el C++ solo ejecuta.
- Audio mínimo en Agentica: lector WAV 16 kHz mono (puro, `std/binary` +
  `File`), luego captura de micrófono en streaming básico (fichero primero,
  micro después — decisión 2026-11-22).
- Modelos small/medium (español); pesos `gguf`/ggml descargados, no en git.
  Prebuilt por plataforma como torch.
- Criterio de salida: `whisper_test` transcribe un wav de referencia en
  español y el ejemplo micrófono→texto funciona.
- Valida el patrón native-gen + C++ pesado + prebuilt antes del torch.

**Vía B — Whisper puro sobre `tensor`/`nn`/`hub` (el examen final).**
- `nn/` se diseña driven por transformers: los requisitos genéricos de un
  cliente de inferencia están en `tensor/DESIGN.md` §5 (`matmul`, `softmax`,
  `layer_norm`, multi-head attention, `gelu/silu`, `conv1d` para un stem
  acústico, embeddings posicionales, `argmax`/greedy + beam search,
  temperatura). Si el cliente necesita un op que falte, entra en el contrato
  antes de congelarlo.
- Frontend log-mel (80 bins, ventanas 30 s) en Agentica puro donde sea
  viable, nativo solo si el spike lo exige (STFT/mel es el primer candidato
  a `compute def`).
- `hub/` prioriza **pesos Whisper en safetensors** (sharding + índice JSON)
  como primer caso real, además del MLP de juguete.
- Criterio de salida: el mismo wav de la vía A transcrito por el stack puro
  (aunque sea más lento y solo en reference/torch CPU al principio).
- Orden: no bloquea M0–M2; en cuanto `nn/` + `hub/` existan, Whisper-B es
  el ejemplo que los prueba. `train/` sigue después (Whisper inferencia no
  necesita entrenar).

## 5. Set mínimo de ops del backend (~guía, se congela en fase 0)

creación (`zeros/ones/empty/full/like`, `rand` con semilla) ·
elementwise (`add/sub/mul/div`, `exp/log/sqrt/pow`, `relu/sigmoid/tanh`,
comparaciones, `where/clamp`) · reducciones (`sum/mean/max/argmax`) ·
linalg (`matmul`, `dot`, transpose) · forma (`reshape/squeeze/unsqueeze`,
`slice/concat/stack`, `broadcast`) · conv/pooling básico · random
(`randn`, `randint`, split de claves) · movimiento (`to_device`,
`to_dtype`, `copy`). Todo lo demás (grad, vmap, jit) vive en Agentica.

## 6. Testing y CI

- **Oráculo**: la referencia decide qué es correcto; cada backend pasa la
  misma suite de conformidad (valores + shapes + dtypes + errores).
- Unit tests (`*_test.agc`) para lo puro; `.feature` + steps para
  entrenamiento e inferencia; `doc-examples` para los bloques de docs.
- Headers nativos generados comprometidos + `--check` en CI (nada stale).
- Benchmarks fuera de `_build/bytecode` cuando se mida (medir `run` sobre
  `.agcb` compilado, no `agentica FILE.agc`, o se mide el frontend).
- CI por paquete con su `AGENTICA_VERSION`; cuando haya releases, solo los
  tests del paquete cambiado (issue pendiente en `packages/`).

## 7. Riesgos

1. **Tamaño/build de libtorch** (CPU primero mitiga; prebuilt obligatorio).
2. **GC vs intermedios** (bloquea el diseño del loop; depende de arenas).
3. **PJRT es un proyecto en sí mismo** (por eso va en fase 6, aislado).
4. **Cobertura de dtypes/ops** (cada op × dtype × backend multiplica la
   matriz de tests; la conformidad compartida la contiene).
5. **Divergencia con `packages/`** (native-gen evoluciona allí; fijar tag
   por `agentica.toml`, p. ej. `native-gen-0.4.0`, y subirlo a propósito).

## 8. Hitos y criterio de "hecho"

| Hito | Contiene | Hecho cuando |
|---|---|---|
| M0 spike | tensor crear + matmul + grad juguete + 1 paso SGD en reference y torch CPU | gaps FFI listados, API mínima congelada |
| M1 tensor | tipos, ~60 ops, traza+grad, referencia, conformidad | conformidad verde en referencia |
| M2 torch | capa C++ vía native-gen dentro de `tensor/`, libtorch `optional` | conformidad verde en el backend libtorch (18/45 primitivas) y en la referencia; 27 primitivas pendientes; prebuilt CPU y el job de CI con libtorch, pendientes |
| M3 nn | capas + inferencia MLP | inferencia con `nn` + torch, ejemplo runnable |
| M4 train | losses + SGD/Adam + loop | MLP entrena end-to-end, loss baja, test lo prueba |
| M5 hub | safetensors + descarga HF | pesos reales cargados en `nn` |
| M6 xla | PJRT compila el grafo | mismo programa corre eager y JIT |
| M7 candle | backend eager Rust | conformidad verde en candle |
| M8 ggml | inferencia transformers | fuera del MVP, solo diseño hasta M5 |
| WA whisper binding | `whisper/` sobre whisper.cpp + wav 16 kHz + micro, small/medium ES | transcribe el wav de referencia; valida native-gen + prebuilt |
| WB whisper puro | mismo wav con `tensor`/`nn`/`hub` (reference/torch CPU) | paridad funcional con WA en el caso de prueba |

## 9. Próximos pasos

1. Verificar estado de agentica#1091 (arenas) y fijar Agentica mínima.
2. Cerrar las 4 preguntas abiertas de ml#1 con el owner **antes** del
   diseño detallado (tal como pide el issue).
3. **Vía A ya**: `cli.agc new whisper` en este repo contra `../whisper.cpp`
   (solo necesita native-gen + un `whisper.cpp` compilable; no depende del
   diseño de `tensor/`). Lector WAV 16 kHz en Agentica puro + test con wav
   de referencia ES + modelo small/medium.
4. Diseño de `tensor/` **ya escrito**: `tensor/DESIGN.md` (contrato y
   decisiones abiertas). Lo siguiente es el spike M0 y el detalle por firma.
5. `cli.agc new torch` en este repo, header inicial, spike M0.
6. Fijar `AGENTICA_VERSION` + tag de native-gen y abrir el workspace.

---

*Notas de nomenclatura: "glm" del encargo se lee como **ggml**
(el tensor engine en C tras llama.cpp). Si era otra cosa (p. ej. GLM de
OpenGL o un modelo concreto), decirlo y se ajusta la fase 8.*
