# note — medidas y límites

Las medidas de una iGPU y los límites conocidos de CI. Lo hecho está en
`CHANGELOG.md` y en los PRs; el orden de trabajo, en `roadmap.md`; el
contrato del tensor, en `tensor/DESIGN.md`.

## Medidas que importan (esta máquina: Ryzen 5 3550H, Vega iGPU RADV, AVX2+FMA, **sin AVX512**)

- **2.00 ms por grafo** despachado a Vulkan. Por eso la iGPU **solo gana
  con ≥~1 GFLOP** (512×2048: 3.1×, 1024³: 1.8× frente a 8 hilos AVX2) y
  **pierde por debajo** (256³: 2.9× más lenta).
- `mul_mat` de ggml CPU es 1.5–3.4× más lento que el BLAS de libtorch.
- Consecuencia de diseño: **Vulkan no puede ser un backend op-por-op**;
  pide batching de grafo.
- AVX512 no es alcanzable aquí: el techo real es AVX2.
- **Un paso compilado a un grafo ggml sí sirve en la iGPU** (spike
  ggml-graph, MLP 784→512→512→10, SGD). Con `mul_mat` F32 (el default):
  4.5–4.6 / 7.2–7.3 / 18.9 / 68 ms por paso a batch 64 / 256 / 1024 / 4096.
  Contra el mejor libtorch C++ en CPU de cada corrida (4 u 8 hilos; máquina
  ociosa / cargada) gana **1.2–1.5× a 256, 1.5–2.2× a 1024 y 1.8–2.1× a
  4096**, y pierde 1.4–1.6× a 64. Con fp16 (opcional): 1.5–1.9×, 2.1–2.9× y
  2.7–3.2×, pero un tensor de gradientes sale con un 10 % de error a esas
  formas. Ninguna op del paso falta en Vulkan. Trazar, derivar y bajar
  cuesta 2–6 ms una vez; luego un paso es una llamada FFI.

## 5. Límites conocidos (no son bugs, son el trato)

- **CI no compila el C++ de libtorch**: no se puede bajar un binario de
  473 MB por push. Un job verde prueba `check`, `fmt`, el header
  sincronizado y el camino "módulo ausente"; un error de compilación o una
  deriva de la API aparece como un warning en un job verde. La señal es
  `warning: native module torch not built`.
- **El runner solo imprime PASS**: un test que hace SKIP no deja rastro
  visible, así que los skips deben ser estructurales (aserción del estado
  ausente), no un `puts` y `ok`.
- **Vulkan no está en CI** (no hay driver pineable): el CI compila ggml solo
  CPU y corre allí los tests de `Backend.Ggml(Device.Cpu)`; los de Vulkan
  corren en local y, sin dispositivo, afirman el estado ausente.
