/* Spike ml#80: GSPMD on a PJRT plugin. x[B,K] . w[K,N] on 1 device, then w split by columns and by rows over 2.
   cc -O2 -I ml/tensor/native gspmd.c -ldl -o gspmd && ./gspmd PLUGIN [B K N] */
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "pjrt_c_api.h"

static const PJRT_Api *api;
static PJRT_Client *client;
static PJRT_Device *const *devs;
static size_t ndev;

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

static void check(PJRT_Error *e, const char *what) {
  if (e == NULL) return;
  PJRT_Error_Message_Args m = {.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE, .error = e};
  api->PJRT_Error_Message(&m);
  fprintf(stderr, "%s: %.*s\n", what, (int)m.message_size, m.message);
  exit(1);
}

static void await(PJRT_Event *ev) {
  if (ev == NULL) return;
  PJRT_Event_Await_Args w = {.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE, .event = ev};
  check(api->PJRT_Event_Await(&w), "await");
  PJRT_Event_Destroy_Args d = {.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE, .event = ev};
  api->PJRT_Event_Destroy(&d);
}

static PJRT_Buffer *upload(const float *data, int64_t r, int64_t c, size_t dev) {
  int64_t dims[2] = {r, c};
  PJRT_Client_BufferFromHostBuffer_Args a = {.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE,
                                             .client = client, .data = data, .type = PJRT_Buffer_Type_F32,
                                             .dims = dims, .num_dims = 2,
                                             .host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
                                             .device = devs[dev]};
  check(api->PJRT_Client_BufferFromHostBuffer(&a), "upload");
  await(a.done_with_host_buffer);
  return a.buffer;
}

static float *download(PJRT_Buffer *b, size_t *n) {
  PJRT_Buffer_Dimensions_Args dm = {.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE, .buffer = b};
  check(api->PJRT_Buffer_Dimensions(&dm), "dims");
  *n = 1;
  for (size_t k = 0; k < dm.num_dims; k++) *n *= dm.dims[k];
  float *dst = malloc(*n * 4);
  PJRT_Buffer_ToHostBuffer_Args a = {.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE, .src = b, .dst = dst,
                                     .dst_size = *n * 4};
  check(api->PJRT_Buffer_ToHostBuffer(&a), "download");
  await(a.event);
  return dst;
}

static PJRT_LoadedExecutable *compile(const char *text, int parts, double *secs) {
  /* CompileOptionsProto{executable_build_options{num_replicas: 1, num_partitions: parts, use_spmd_partitioning}} */
  /* parts 2: + use_spmd_partitioning, device_assignment {replica_count 1, computation_count 2, {0}, {1}} and
     use_shardy_partitioner (19); without the device assignment compile fails (PartitionId) */
  static const char one[] = {0x1a, 0x04, 0x20, 0x01, 0x28, 0x01};
  static const char two[] = {0x1a, 0x19, 0x20, 0x01, 0x28, 0x02, 0x30, 0x01, 0x4a, 0x0e, 0x08, 0x01, 0x10, 0x02, 0x1a,
                             0x03, 0x0a, 0x01, 0x00, 0x1a, 0x03, 0x0a, 0x01, 0x01, (char)0x98, 0x01, 0x01};
  const char *opts = parts == 1 ? one : two;
  size_t optn = parts == 1 ? sizeof one : sizeof two;
  static char full[16384];
  char path[64];
  snprintf(path, sizeof path, "opts%d.bin", parts);
  FILE *f = getenv("JAX_OPTS") ? fopen(path, "rb") : NULL;
  size_t olen = optn;
  const char *ob = opts;
  if (f != NULL) {
    olen = fread(full, 1, sizeof full, f);
    fclose(f);
    ob = full;
  }
  PJRT_Program p = {.struct_size = PJRT_Program_STRUCT_SIZE, .code = (char *)text, .code_size = strlen(text),
                    .format = "mlir", .format_size = 4};
  PJRT_Client_Compile_Args c = {.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE, .client = client, .program = &p,
                                .compile_options = ob, .compile_options_size = olen};
  double t = now();
  check(api->PJRT_Client_Compile(&c), "compile");
  *secs = now() - t;
  return c.executable;
}

/* Runs `exe` on `nd` devices with `nargs` args each, `reps` times; the median ms; the outputs of the last run. */
static double run(PJRT_LoadedExecutable *exe, size_t nd, PJRT_Buffer *args[][2], size_t nargs, PJRT_Buffer *outs[],
                  int reps) {
  double ms[64];
  for (int r = 0; r < reps; r++) {
    PJRT_Buffer *const *al[2] = {args[0], args[1]};
    PJRT_Buffer *o0[1], *o1[1];
    PJRT_Buffer **ol[2] = {o0, o1};
    PJRT_Event *done[2] = {NULL, NULL};
    PJRT_ExecuteOptions eo = {.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE};
    PJRT_LoadedExecutable_Execute_Args ex = {.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE,
                                             .executable = exe, .options = &eo, .argument_lists = al,
                                             .num_devices = nd, .num_args = nargs, .output_lists = ol,
                                             .device_complete_events = done};
    double t = now();
    check(api->PJRT_LoadedExecutable_Execute(&ex), "execute");
    for (size_t d = 0; d < nd; d++) await(done[d]);
    ms[r] = (now() - t) * 1000;
    for (size_t d = 0; d < nd; d++) {
      if (r == reps - 1) {
        outs[d] = ol[d][0];
      } else {
        PJRT_Buffer_Destroy_Args bd = {.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE, .buffer = ol[d][0]};
        api->PJRT_Buffer_Destroy(&bd);
      }
    }
  }
  for (int i = 0; i < reps; i++)
    for (int j = i + 1; j < reps; j++)
      if (ms[j] < ms[i]) { double s = ms[i]; ms[i] = ms[j]; ms[j] = s; }
  return ms[reps / 2];
}

static double gap(const float *a, const float *b, size_t n) {
  double worst = 0, scale = 1e-6;
  for (size_t i = 0; i < n; i++) {
    worst = fmax(worst, fabs(a[i] - b[i]));
    scale = fmax(scale, fabs(b[i]));
  }
  return worst / scale;
}

int main(int argc, char **argv) {
  int64_t B = argc > 4 ? atoll(argv[2]) : 8, K = argc > 4 ? atoll(argv[3]) : 16, N = argc > 4 ? atoll(argv[4]) : 32;
  const int REPS = 31;
  setvbuf(stdout, NULL, _IONBF, 0);
  void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (lib == NULL) return fprintf(stderr, "%s\n", dlerror()), 1;
  api = ((const PJRT_Api *(*)(void))dlsym(lib, "GetPjrtApi"))();
  PJRT_Plugin_Initialize_Args init = {.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE};
  printf("api %d.%d\n", api->pjrt_api_version.major_version, api->pjrt_api_version.minor_version);
  check(api->PJRT_Plugin_Initialize(&init), "init");
  printf("initialized\n");
  PJRT_Client_Create_Args cc = {.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE};
  check(api->PJRT_Client_Create(&cc), "client");
  client = cc.client;
  printf("client\n");
  PJRT_Client_AddressableDevices_Args ad = {.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE,
                                            .client = client};
  check(api->PJRT_Client_AddressableDevices(&ad), "devices");
  devs = ad.addressable_devices;
  ndev = ad.num_addressable_devices;
  printf("devices: %zu\n", ndev);
  for (size_t d = 0; d < ndev; d++) {
    PJRT_Device_GetDescription_Args gd = {.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE,
                                          .device = devs[d]};
    check(api->PJRT_Device_GetDescription(&gd), "description");
    PJRT_DeviceDescription_Kind_Args k = {.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE,
                                          .device_description = gd.device_description};
    check(api->PJRT_DeviceDescription_Kind(&k), "kind");
    PJRT_DeviceDescription_DebugString_Args ds = {.struct_size = PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE,
                                                  .device_description = gd.device_description};
    check(api->PJRT_DeviceDescription_DebugString(&ds), "debug");
    printf("  %zu: kind %s, %.*s\n", d, k.device_kind, (int)ds.debug_string_size, ds.debug_string);
    if (getenv("NO_STATS")) continue;
    PJRT_Device_MemoryStats_Args ms = {.struct_size = PJRT_Device_MemoryStats_Args_STRUCT_SIZE, .device = devs[d]};
    PJRT_Error *e = api->PJRT_Device_MemoryStats(&ms);
    printf("  %zu: kind %s, %.*s, memory limit %lld bytes%s\n", d, k.device_kind, (int)ds.debug_string_size,
           ds.debug_string, ms.bytes_limit_is_set ? (long long)ms.bytes_limit : -1LL,
           e ? " (MemoryStats unsupported)" : "");
  }
  if (ndev < 2) return printf("one device: no GSPMD test\n"), 0;

  float *x = malloc(B * K * 4), *w = malloc(K * N * 4);
  for (int64_t i = 0; i < B * K; i++) x[i] = (float)((i % 13) - 6) / 13.0f;
  for (int64_t i = 0; i < K * N; i++) w[i] = (float)((i % 7) - 3) / 7.0f;
  char text[4096];
  double secs;
  PJRT_Buffer *args[2][2], *outs[2];
  size_t n;
  float *want = NULL;

  snprintf(text, sizeof text,
           "module @one {\n func.func public @main(%%x: tensor<%ldx%ldxf32>, %%w: tensor<%ldx%ldxf32>) -> tensor<%ldx%ldxf32> {\n"
           "  %%0 = stablehlo.dot_general %%x, %%w, contracting_dims = [1] x [0] : (tensor<%ldx%ldxf32>, tensor<%ldx%ldxf32>) -> tensor<%ldx%ldxf32>\n"
           "  return %%0 : tensor<%ldx%ldxf32>\n }\n}\n",
           B, K, K, N, B, N, B, K, K, N, B, N, B, N);
  if (getenv("SKIP_ONE") && getenv("SKIP_ONE")[0] == '2') goto col_part;
  PJRT_LoadedExecutable *one = compile(text, 1, &secs);
  if (getenv("SKIP_ONE")) goto col_part;
  args[0][0] = upload(x, B, K, 0);
  args[0][1] = upload(w, K, N, 0);
  double t1 = run(one, 1, args, 2, outs, REPS);
  want = download(outs[0], &n);
  printf("one device: compile %.3f s, run %.3f ms (B=%ld K=%ld N=%ld)\n", secs, t1, B, K, N);

col_part:
  if (want == NULL) want = calloc(B * N, 4);
  /* Columns: w[K, N/2] per device, x replicated, out[B, N/2] per device. */
  snprintf(text, sizeof text,
           "module @col attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {\n sdy.mesh @mesh = <[\"m\"=2]>\n"
           " func.func public @main(%%x: tensor<%ldx%ldxf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %%w: tensor<%ldx%ldxf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {\"m\"}]>})"
           " -> (tensor<%ldx%ldxf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {\"m\"}]>}) {\n"
           "  %%0 = stablehlo.dot_general %%x, %%w, contracting_dims = [1] x [0] : (tensor<%ldx%ldxf32>, tensor<%ldx%ldxf32>) -> tensor<%ldx%ldxf32>\n"
           "  return %%0 : tensor<%ldx%ldxf32>\n }\n}\n",
           B, K, K, N, B, N, B, K, K, N, B, N, B, N);
  printf("compiling col\n");
  FILE *tf = getenv("COL_TEXT") ? fopen(getenv("COL_TEXT"), "r") : NULL;
  if (tf != NULL) {
    size_t got = fread(text, 1, sizeof text - 1, tf);
    text[got] = 0;
    fclose(tf);
  }
  PJRT_LoadedExecutable *col = compile(text, 2, &secs);
  printf("compiled col\n");
  float *wc = malloc(K * N * 4);
  for (int d = 0; d < 2; d++) {
    for (int64_t k = 0; k < K; k++) memcpy(wc + d * K * (N / 2) + k * (N / 2), w + k * N + d * (N / 2), (N / 2) * 4);
    args[d][0] = upload(x, B, K, d);
    args[d][1] = upload(wc + d * K * (N / 2), K, N / 2, d);
  }
  printf("running col\n");
  double t2 = run(col, 2, args, 2, outs, REPS);
  double worst = 0;
  for (int d = 0; d < 2; d++) {
    float *got = download(outs[d], &n), *part = malloc(n * 4);
    for (int64_t b = 0; b < B; b++) memcpy(part + b * (N / 2), want + b * N + d * (N / 2), (N / 2) * 4);
    worst = fmax(worst, gap(got, part, n));
  }
  printf("column split: compile %.3f s, run %.3f ms, gap %.2g\n", secs, t2, worst);

  /* Rows: x[B, K/2] and w[K/2, N] per device, out replicated (an all-reduce). */
  snprintf(text, sizeof text,
           "module @row attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {\n sdy.mesh @mesh = <[\"m\"=2]>\n"
           " func.func public @main(%%x: tensor<%ldx%ldxf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {\"m\"}]>}, %%w: tensor<%ldx%ldxf32> {sdy.sharding = #sdy.sharding<@mesh, [{\"m\"}, {}]>})"
           " -> (tensor<%ldx%ldxf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {\n"
           "  %%0 = stablehlo.dot_general %%x, %%w, contracting_dims = [1] x [0] : (tensor<%ldx%ldxf32>, tensor<%ldx%ldxf32>) -> tensor<%ldx%ldxf32>\n"
           "  return %%0 : tensor<%ldx%ldxf32>\n }\n}\n",
           B, K, K, N, B, N, B, K, K, N, B, N, B, N);
  FILE *rf = getenv("ROW_TEXT") ? fopen(getenv("ROW_TEXT"), "r") : NULL;
  if (rf != NULL) {
    size_t got = fread(text, 1, sizeof text - 1, rf);
    text[got] = 0;
    fclose(rf);
  }
  PJRT_LoadedExecutable *row = compile(text, 2, &secs);
  float *xr = malloc(B * K * 4);
  for (int d = 0; d < 2; d++) {
    for (int64_t b = 0; b < B; b++) memcpy(xr + d * B * (K / 2) + b * (K / 2), x + b * K + d * (K / 2), (K / 2) * 4);
    args[d][0] = upload(xr + d * B * (K / 2), B, K / 2, d);
    args[d][1] = upload(w + d * (K / 2) * N, K / 2, N, d);
  }
  double t3 = run(row, 2, args, 2, outs, REPS);
  worst = 0;
  for (int d = 0; d < 2; d++) worst = fmax(worst, gap(download(outs[d], &n), want, n));
  printf("row split: compile %.3f s, run %.3f ms, gap %.2g\n", secs, t3, worst);
  return 0;
}
