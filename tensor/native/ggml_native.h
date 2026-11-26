/* The C side of `extern module "ggml"` in ggml_native.agc, made by native-gen.
   Do not edit it: change the .agc and make it again. */
#ifndef GGML_NATIVE_H
#define GGML_NATIVE_H

#include "agentica_native.h"

#if AN_API_MAJOR != 1
#error "ggml_native.h needs version 1 of the native API (agentica_native.h)"
#endif
#if AN_API_MINOR < 2
#error "ggml_native.h needs version 1.2 or later of the native API (agentica_native.h)"
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* A String or Bytes argument, borrowed for the call: copy it to keep it. String data is UTF-8 with no NUL at its
   end. Every header of this version of native-gen has these two, so one file can include several. */
#ifndef NATIVE_GEN_SLICES_1
#define NATIVE_GEN_SLICES_1
typedef struct {
  const char *data;
  size_t len;
} ng_str;
typedef struct {
  const uint8_t *data;
  size_t len;
} ng_buf;
#endif

/* The rows of the resource table: the TYPE of an_resource_new and an_get_resource. */
enum { GGML_RESOURCE_TensorHandle = 0, GGML_RESOURCE_Lease = 1, GGML_RESOURCE_Pin = 2 };

/* One tensor in a ggml backend buffer of its device: the state owns a
   ggml context holding the tensor's metadata and the buffer holding its
   data, both freed by the free hook. Nothing else points at them.

   Define it, and GGML_TensorHandle_FREE (a void (*)(void *state), or NULL), before GGML_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct ggml_TensorHandle_state;
static inline struct ggml_TensorHandle_state *ggml_TensorHandle_new(an_env *env, an_term *out) {
  return (struct ggml_TensorHandle_state *)an_resource_new(env, GGML_RESOURCE_TensorHandle, out);
}
static inline struct ggml_TensorHandle_state *ggml_TensorHandle_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, GGML_RESOURCE_TensorHandle, &state) ? (struct ggml_TensorHandle_state *)state : NULL;
}

/* One compiled graph of the structural cache, held by one evaluation. The
   graph itself lives in the C cache for the process; the lease marks it
   in use, and its free hook gives it back (or drops it if it never
   compiled). So two tasks never share one graph's leaves at once.

   Define it, and GGML_Lease_FREE (a void (*)(void *state), or NULL), before GGML_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct ggml_Lease_state;
static inline struct ggml_Lease_state *ggml_Lease_new(an_env *env, an_term *out) {
  return (struct ggml_Lease_state *)an_resource_new(env, GGML_RESOURCE_Lease, out);
}
static inline struct ggml_Lease_state *ggml_Lease_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, GGML_RESOURCE_Lease, &state) ? (struct ggml_Lease_state *)state : NULL;
}

/* Keeps the compiled graph of `lease` in the structural cache for as long
   as the `Pin` lives, however many other graphs the cache takes in: the
   graph is exempt from eviction. The pin ends when its owner closes it or
   ends; the graph then stays in the cache as any other. The lease must hold
   a compiled graph that has a slot of the cache.

   Define it, and GGML_Pin_FREE (a void (*)(void *state), or NULL), before GGML_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct ggml_Pin_state;
static inline struct ggml_Pin_state *ggml_Pin_new(an_env *env, an_term *out) {
  return (struct ggml_Pin_state *)an_resource_new(env, GGML_RESOURCE_Pin, out);
}
static inline struct ggml_Pin_state *ggml_Pin_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, GGML_RESOURCE_Pin, &state) ? (struct ggml_Pin_state *)state : NULL;
}

/* The Lists, Maps, Optionals and Results. A List or Map that is read is a cursor: _next gives 1 for an item, 0 at
   the end, and -1 when the item does not decode: the call has then failed with the path of the item in its own
   List or Map (an item of a List or Map read from another has the path from the inner one, `List[Float].1`),
   after `argument N: ` when the cursor came from an argument (`argument` is then N; it is 0 after _open), and
   every later _next gives -1 too, as after a failed _open. A builder makes one: _begin, then _push (_put for a
   Map), then _end. Cursors and builders hold terms: keep them in C locals, for this call only. An Optional or
   Result that is read is a struct, whose value is zero when there is none and whose error is borrowed as a String
   argument is; _get reads one held as a term. One that is made has _some or _ok (an_none and an_error make the
   others). */

typedef struct {
  an_term term;
  an_list_cursor cursor;
  size_t len;
  size_t at;
  bool failed;
  int argument; /* 1 + the index of the argument of the call it was opened for, or 0 (_open) */
} ggml_list_Int;
bool ggml_list_Int_open(an_env *env, an_term t, ggml_list_Int *out);
int ggml_list_Int_next(an_env *env, ggml_list_Int *list, int64_t *out);

typedef struct {
  an_term term;
  an_list_cursor cursor;
  size_t len;
  size_t at;
  bool failed;
  int argument; /* 1 + the index of the argument of the call it was opened for, or 0 (_open) */
} ggml_list_Float;
bool ggml_list_Float_open(an_env *env, an_term t, ggml_list_Float *out);
int ggml_list_Float_next(an_env *env, ggml_list_Float *list, double *out);

an_term ggml_result_TensorHandle_ok(an_env *env, an_term value);

typedef struct {
  an_list_builder builder;
} ggml_list_Float_builder;
void ggml_list_Float_begin(an_env *env, ggml_list_Float_builder *b, size_t hint);
void ggml_list_Float_push(an_env *env, ggml_list_Float_builder *b, double value);
an_term ggml_list_Float_end(an_env *env, ggml_list_Float_builder *b);

an_term ggml_result_list_Float_ok(an_env *env, an_term value);

an_term ggml_result_Void_ok(an_env *env, an_term value);

an_term ggml_result_Lease_ok(an_env *env, an_term value);

an_term ggml_result_Int_ok(an_env *env, int64_t value);

an_term ggml_result_Pin_ok(an_env *env, an_term value);

/* How many Vulkan devices this build of ggml sees, not counting a CPU
   device (a software rasterizer): 0 when ggml was built without its Vulkan
   backend (`tools/build-engine.sh cpu`). It opens no device. */
an_term ggml_vulkan_devices(an_env *env);

/* A tensor of the row-major `shape` (rank 4 at most) on `device`, holding
   `data`, one value per element. */
an_term ggml_upload(an_env *env, int64_t device, ggml_list_Int shape, ggml_list_Float data, bool int32);

/* A tensor of the row-major `shape` on `device` from the `format` (a code of
   `Types.file_code`) values that start at byte `offset` of the file `path`,
   held as `held` says (a code of `Types.held_code`): F32, F16 (a value
   beyond 65504 is an error) or Q8_0, Q4_K or Q6_K, the blocks of a
   range of that type as they are.
   C reads the file itself, so no value crosses as a list; a range past the
   end of the file is an error. An F16 tensor converts the values a chunk at
   a time. */
an_term ggml_load(an_env *env, int64_t device, ng_str path, int64_t offset, ggml_list_Int shape, int64_t format, int64_t held);

/* The same from `data`, the bytes of the range that Agentica read
   (`TENSOR_LOAD=agentica`): the call reads them in place and widens them into
   the buffer. `data` must be exactly the range, or the call is an error. */
an_term ggml_from_bytes(an_env *env, int64_t device, ng_buf data, ggml_list_Int shape, int64_t format, int64_t held);

/* A tensor of `shape` on `device` with every element `value`, filled in C,
   so a large fill never crosses as a list. */
an_term ggml_filled(an_env *env, int64_t device, ggml_list_Int shape, double value, bool int32);

/* True when `t` lives in host memory, made for the CPU backend (`device` -1): an output of a graph that runs on a
   Vulkan device and a CPU together may be one. */
an_term ggml_on_cpu(an_env *env, struct ggml_TensorHandle_state *t);

/* Host readback: the row-major values as doubles, which the caller tags
   with the tensor's dtype. */
an_term ggml_readback(an_env *env, struct ggml_TensorHandle_state *t);

/* Appends the values of the `F32` tensor `t`, row-major and little-endian, to the existing file `path`: C writes
   them, so no value crosses as a list. */
an_term ggml_append_file(an_env *env, struct ggml_TensorHandle_state *t, ng_str path);

/* A copy of `t` on `device`, of the same shape and storage. */
an_term ggml_transfer(an_env *env, struct ggml_TensorHandle_state *t, int64_t device);

/* A compiled graph cached under `key` on `device` that no one holds, or a
   new empty one with room for `capacity` nodes, to build with `node` and
   `compile`. A new graph with `keep` false takes no cache slot: it is
   dropped when its lease is freed. */
an_term ggml_acquire(an_env *env, int64_t device, ng_str key, int64_t capacity, bool keep);

/* True when the leased graph is compiled: a cache hit. */
an_term ggml_ready(an_env *env, struct ggml_Lease_state *lease);

/* Appends one node and answers its index: `op` is a code of
   `native/ggml_graph.c`, `args` earlier indices, `shape` the row-major
   shape the trace recorded, `ints` the op's integer parameters. */
an_term ggml_node(an_env *env, struct ggml_Lease_state *lease, int64_t op, ggml_list_Int args, ggml_list_Int shape, ggml_list_Int ints);

/* Builds the graph of `outputs`, allocates its fills and checks every
   node against the backend. A failed compile leaves the graph unusable. */
an_term ggml_compile(an_env *env, struct ggml_Lease_state *lease, ggml_list_Int outputs);

/* Makes `t` the leaf `slot` for the next run (leaves count in the order
   they were sent): an F32 tensor is read in place, an I32 one is converted
   into the leaf's own memory when the run starts. The graph holds `t`'s
   memory until the lease ends, so a `t` that is closed meanwhile does not
   leave the run without its data. It touches no device and never waits. */
an_term ggml_bind(an_env *env, struct ggml_Lease_state *lease, int64_t slot, struct ggml_TensorHandle_state *t);

/* A fresh tensor that the next run writes output `output` into, stored as
   I32 when `int32` says so. */
an_term ggml_output(an_env *env, struct ggml_Lease_state *lease, int64_t output, bool int32);

/* Makes `t`, a tensor that exists, the tensor that the next run writes
   output `output` into, in place of a fresh one: an F32 tensor of the same
   element count is written directly. `t` must not be a leaf of this run. */
an_term ggml_output_into(an_env *env, struct ggml_Lease_state *lease, int64_t output, struct ggml_TensorHandle_state *t);

an_term ggml_pin(an_env *env, struct ggml_Lease_state *lease);

/* The number that names `pin` among the pins of the process, for `resume`
   and `keep`: 0 for a pin that holds nothing. */
an_term ggml_pin_id(an_env *env, struct ggml_Pin_state *pin);

/* True when the leased graph still has the constants of the program of the
   pin `pin` (its `pin_id`) bound, kept by `keep` after an earlier run: the
   caller then binds only what every run binds (the inputs, and a constant
   that is converted). Otherwise the graph lets go of what it kept, for
   whatever program, and the caller binds every leaf. A kept tensor whose
   handle is closed, another program that ran on the graph, a pin that ended
   and a pin of another graph all answer false. */
an_term ggml_resume(an_env *env, struct ggml_Lease_state *lease, int64_t pin);

/* Keeps the leaves `slots` bound after the run, as the constants of the
   program of the pin `pin`, until that pin ends, one of the tensors is closed
   or another program runs on the graph. The graph holds their memory
   meanwhile. It keeps every leaf of `slots` or none: nothing is kept when
   one of them is not bound, is converted (an `I32` tensor) or is written in
   place, or unless `pin` is a live pin of the leased graph. */
an_term ggml_keep(an_env *env, struct ggml_Lease_state *lease, int64_t pin, ggml_list_Int slots);

/* How many leaves the leased graph keeps bound, for a test to read. */
an_term ggml_kept(an_env *env, struct ggml_Lease_state *lease);

/* Sets the fills (one value each, in the order they were sent) and runs
   the graph into the tensors `output` made; every leaf and output must be
   set since the last run. An error when an output is not finite. */
an_term ggml_run(an_env *env, struct ggml_Lease_state *lease, ggml_list_Float fills);

/* Writes the `count` values of `src_format` (a code of `Types.file_code`) that start at byte `src_offset` of the
   file `src` into the existing file `dst` at byte `dst_offset`, as `dst_format`. The same format is a byte copy;
   F32 to F16 rounds to nearest even and refuses a value beyond 65504, and to BF16 it rounds to nearest even. C
   does it in chunks on several threads, so no value crosses as a list. Past the end of `dst` the file grows. */
an_term ggml_convert_file(an_env *env, ng_str src, int64_t src_offset, int64_t src_format, int64_t count, ng_str dst, int64_t dst_offset, int64_t dst_format);

/* How many graphs this process has compiled: the cache's misses. */
an_term ggml_compiles(an_env *env);

/* The threads of the CPU backend (or the ones it will run on): `TENSOR_THREADS`, else the physical cores. */
an_term ggml_cpu_threads(an_env *env);

/* Chooses fp16 (`true`) or F32 (`false`, the default) `mul_mat` for every
   Vulkan device this process opens. ggml reads it once, when it first
   enumerates its devices, so a different choice after any other call of
   this module (a CPU one included) is an error. */
an_term ggml_vulkan_fp16(an_env *env, bool enabled);

#ifdef __cplusplus
}
#endif

#endif

/* Outside the include guard, so a file that included the header before it defined the switch still gets this
   part; its own guard keeps it to one copy. */
#if defined(GGML_NATIVE_IMPLEMENTATION) && !defined(GGML_NATIVE_IMPLEMENTATION_DONE)
#define GGML_NATIVE_IMPLEMENTATION_DONE
/* Declared with C linkage, so a module built as C++ exports it under its C name. */
#ifdef __cplusplus
extern "C" {
#endif
AN_EXPORT const an_module *agentica_native_init(const an_api *api);
#ifdef __cplusplus
}
#endif

#ifndef NATIVE_GEN_FAIL_1
#define NATIVE_GEN_FAIL_1
#include <stdio.h>
#include <string.h>

/* Where a decode failed: the path from the outer value in, and the type that was expected there. */
typedef struct {
  char path[192];
  size_t len;
  bool cut;
  const char *expected;
} ng_fail;

static inline bool ng_expected(ng_fail *fail, const char *expected) {
  fail->path[0] = '\0';
  fail->len = 0;
  fail->cut = false;
  fail->expected = expected;
  return false;
}

/* Puts SEGMENT in front of the path. The first one that does not fit becomes "..." and ends the path, which
   keeps its inner part. */
static inline bool ng_at(ng_fail *fail, const char *segment) {
  size_t n = strlen(segment), dot = fail->len > 0 ? 1 : 0;
  if (fail->cut) return false;
  if (n + dot + fail->len >= sizeof fail->path) {
    fail->cut = true;
    segment = "...";
    n = 3;
    if (n + dot + fail->len >= sizeof fail->path) return false;
  }
  memmove(fail->path + n + dot, fail->path, fail->len + 1);
  memcpy(fail->path, segment, n);
  if (dot) fail->path[n] = '.';
  fail->len += n + dot;
  return false;
}

static inline bool ng_leaf(ng_fail *fail, const char *expected, const char *segment) {
  ng_expected(fail, expected);
  return ng_at(fail, segment);
}

/* Fails the call with "WHERE TYPE.PATH: expected X". */
static inline bool ng_raise(an_env *env, const char *where, const char *type, ng_fail *fail) {
  char message[256];
  ng_at(fail, type);
  snprintf(message, sizeof message, "%s%s: expected %s", where, fail->path, fail->expected);
  an_raise(env, message);
  return false;
}

/* Fails the call at item INDEX (and PART of it, "key" or "value", or NULL) of a List or Map; -1 for _next. */
static inline int ng_fail_item(an_env *env, ng_fail *fail, const char *type, size_t index, const char *part) {
  char segment[48];
  if (part)
    snprintf(segment, sizeof segment, "%zu.%s", index, part);
  else
    snprintf(segment, sizeof segment, "%zu", index);
  ng_at(fail, segment);
  ng_raise(env, "", type, fail);
  return -1;
}

static inline an_term ng_fail_argument(an_env *env, ng_fail *fail, const char *type, int index) {
  char where[32];
  snprintf(where, sizeof where, "argument %d: ", index + 1);
  ng_raise(env, where, type, fail);
  return an_void(env);
}
#endif

/* A block of its own: the block above is the same in every header since 0.1.0, so a file that includes an older
   header first keeps that copy, and this one still comes in. */
#ifndef NATIVE_GEN_FAIL_ITEM_2
#define NATIVE_GEN_FAIL_ITEM_2
/* As ng_fail_item, with the "argument N: " prefix when ARGUMENT (1 + the index of the argument the cursor was
   opened for) is not 0. */
static inline int ng_fail_cursor_item(an_env *env, ng_fail *fail, const char *type, int argument, size_t index,
                                      const char *part) {
  char segment[48];
  char where[32] = "";
  if (part)
    snprintf(segment, sizeof segment, "%zu.%s", index, part);
  else
    snprintf(segment, sizeof segment, "%zu", index);
  if (argument) snprintf(where, sizeof where, "argument %d: ", argument);
  ng_at(fail, segment);
  ng_raise(env, where, type, fail);
  return -1;
}
#endif

static bool ggml_native_item_Int(an_env *env, an_term x, int64_t *out, ng_fail *fail) {
  return an_get_int(env, x, &(*out)) || ng_expected(fail, "Int");
}

static bool ggml_native_open_list_Int(an_env *env, an_term t, ggml_list_Int *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Int]");
}

bool ggml_list_Int_open(an_env *env, an_term t, ggml_list_Int *out) {
  ng_fail fail;
  return ggml_native_open_list_Int(env, t, out, &fail) || ng_raise(env, "", "List[Int]", &fail);
}

int ggml_list_Int_next(an_env *env, ggml_list_Int *list, int64_t *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (ggml_native_item_Int(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Int]", list->argument, at, NULL);
}

static bool ggml_native_item_Float(an_env *env, an_term x, double *out, ng_fail *fail) {
  return an_get_float(env, x, &(*out)) || ng_expected(fail, "Float");
}

static bool ggml_native_open_list_Float(an_env *env, an_term t, ggml_list_Float *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Float]");
}

bool ggml_list_Float_open(an_env *env, an_term t, ggml_list_Float *out) {
  ng_fail fail;
  return ggml_native_open_list_Float(env, t, out, &fail) || ng_raise(env, "", "List[Float]", &fail);
}

int ggml_list_Float_next(an_env *env, ggml_list_Float *list, double *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (ggml_native_item_Float(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Float]", list->argument, at, NULL);
}

an_term ggml_result_TensorHandle_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

void ggml_list_Float_begin(an_env *env, ggml_list_Float_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void ggml_list_Float_push(an_env *env, ggml_list_Float_builder *b, double value) {
  an_list_push(env, &b->builder, an_float(env, value));
}

an_term ggml_list_Float_end(an_env *env, ggml_list_Float_builder *b) { return an_list_end(env, &b->builder); }

an_term ggml_result_list_Float_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggml_result_Void_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggml_result_Lease_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggml_result_Int_ok(an_env *env, int64_t value) {
  return an_ok(env, an_int(env, value));
}

an_term ggml_result_Pin_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

static an_term ggml_native_vulkan_devices(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  (void)argv;
  return ggml_vulkan_devices(env);
}

static an_term ggml_native_upload(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  int64_t a0;
  if (!an_get_int(env, argv[0], &a0)) return an_badarg(env, 0);
  ggml_list_Int a1;
  if (!ggml_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  ggml_list_Float a2;
  if (!ggml_native_open_list_Float(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Float]", 2);
  a2.argument = 3;
  bool a3;
  if (!an_get_bool(env, argv[3], &a3)) return an_badarg(env, 3);
  return ggml_upload(env, a0, a1, a2, a3);
}

static an_term ggml_native_load(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  int64_t a0;
  if (!an_get_int(env, argv[0], &a0)) return an_badarg(env, 0);
  ng_str a1;
  if (!an_get_string(env, argv[1], &a1.data, &a1.len)) return an_badarg(env, 1);
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  ggml_list_Int a3;
  if (!ggml_native_open_list_Int(env, argv[3], &a3, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 3);
  a3.argument = 4;
  int64_t a4;
  if (!an_get_int(env, argv[4], &a4)) return an_badarg(env, 4);
  int64_t a5;
  if (!an_get_int(env, argv[5], &a5)) return an_badarg(env, 5);
  return ggml_load(env, a0, a1, a2, a3, a4, a5);
}

static an_term ggml_native_from_bytes(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  int64_t a0;
  if (!an_get_int(env, argv[0], &a0)) return an_badarg(env, 0);
  ng_buf a1;
  if (!an_get_bytes(env, argv[1], &a1.data, &a1.len)) return an_badarg(env, 1);
  ggml_list_Int a2;
  if (!ggml_native_open_list_Int(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 2);
  a2.argument = 3;
  int64_t a3;
  if (!an_get_int(env, argv[3], &a3)) return an_badarg(env, 3);
  int64_t a4;
  if (!an_get_int(env, argv[4], &a4)) return an_badarg(env, 4);
  return ggml_from_bytes(env, a0, a1, a2, a3, a4);
}

static an_term ggml_native_filled(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  int64_t a0;
  if (!an_get_int(env, argv[0], &a0)) return an_badarg(env, 0);
  ggml_list_Int a1;
  if (!ggml_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  double a2;
  if (!an_get_float(env, argv[2], &a2)) return an_badarg(env, 2);
  bool a3;
  if (!an_get_bool(env, argv[3], &a3)) return an_badarg(env, 3);
  return ggml_filled(env, a0, a1, a2, a3);
}

static an_term ggml_native_on_cpu(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_TensorHandle_state *a0 = ggml_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggml_on_cpu(env, a0);
}

static an_term ggml_native_readback(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_TensorHandle_state *a0 = ggml_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggml_readback(env, a0);
}

static an_term ggml_native_append_file(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_TensorHandle_state *a0 = ggml_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  ng_str a1;
  if (!an_get_string(env, argv[1], &a1.data, &a1.len)) return an_badarg(env, 1);
  return ggml_append_file(env, a0, a1);
}

static an_term ggml_native_transfer(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_TensorHandle_state *a0 = ggml_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  return ggml_transfer(env, a0, a1);
}

static an_term ggml_native_acquire(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  int64_t a0;
  if (!an_get_int(env, argv[0], &a0)) return an_badarg(env, 0);
  ng_str a1;
  if (!an_get_string(env, argv[1], &a1.data, &a1.len)) return an_badarg(env, 1);
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  bool a3;
  if (!an_get_bool(env, argv[3], &a3)) return an_badarg(env, 3);
  return ggml_acquire(env, a0, a1, a2, a3);
}

static an_term ggml_native_ready(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggml_ready(env, a0);
}

static an_term ggml_native_node(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  ggml_list_Int a2;
  if (!ggml_native_open_list_Int(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 2);
  a2.argument = 3;
  ggml_list_Int a3;
  if (!ggml_native_open_list_Int(env, argv[3], &a3, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 3);
  a3.argument = 4;
  ggml_list_Int a4;
  if (!ggml_native_open_list_Int(env, argv[4], &a4, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 4);
  a4.argument = 5;
  return ggml_node(env, a0, a1, a2, a3, a4);
}

static an_term ggml_native_compile(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  ggml_list_Int a1;
  if (!ggml_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  return ggml_compile(env, a0, a1);
}

static an_term ggml_native_bind(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  struct ggml_TensorHandle_state *a2 = ggml_TensorHandle_get(env, argv[2]);
  if (!a2) return an_badarg(env, 2);
  return ggml_bind(env, a0, a1, a2);
}

static an_term ggml_native_output(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  bool a2;
  if (!an_get_bool(env, argv[2], &a2)) return an_badarg(env, 2);
  return ggml_output(env, a0, a1, a2);
}

static an_term ggml_native_output_into(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  struct ggml_TensorHandle_state *a2 = ggml_TensorHandle_get(env, argv[2]);
  if (!a2) return an_badarg(env, 2);
  return ggml_output_into(env, a0, a1, a2);
}

static an_term ggml_native_pin(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggml_pin(env, a0);
}

static an_term ggml_native_pin_id(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_Pin_state *a0 = ggml_Pin_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggml_pin_id(env, a0);
}

static an_term ggml_native_resume(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  return ggml_resume(env, a0, a1);
}

static an_term ggml_native_keep(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  ggml_list_Int a2;
  if (!ggml_native_open_list_Int(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 2);
  a2.argument = 3;
  return ggml_keep(env, a0, a1, a2);
}

static an_term ggml_native_kept(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggml_kept(env, a0);
}

static an_term ggml_native_run(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct ggml_Lease_state *a0 = ggml_Lease_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  ggml_list_Float a1;
  if (!ggml_native_open_list_Float(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Float]", 1);
  a1.argument = 2;
  return ggml_run(env, a0, a1);
}

static an_term ggml_native_convert_file(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_str a0;
  if (!an_get_string(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  int64_t a3;
  if (!an_get_int(env, argv[3], &a3)) return an_badarg(env, 3);
  ng_str a4;
  if (!an_get_string(env, argv[4], &a4.data, &a4.len)) return an_badarg(env, 4);
  int64_t a5;
  if (!an_get_int(env, argv[5], &a5)) return an_badarg(env, 5);
  int64_t a6;
  if (!an_get_int(env, argv[6], &a6)) return an_badarg(env, 6);
  return ggml_convert_file(env, a0, a1, a2, a3, a4, a5, a6);
}

static an_term ggml_native_compiles(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  (void)argv;
  return ggml_compiles(env);
}

static an_term ggml_native_cpu_threads(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  (void)argv;
  return ggml_cpu_threads(env);
}

static an_term ggml_native_vulkan_fp16(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  bool a0;
  if (!an_get_bool(env, argv[0], &a0)) return an_badarg(env, 0);
  return ggml_vulkan_fp16(env, a0);
}

static const an_function ggml_native_functions[] = {
  {"vulkan_devices", "() -> Int", ggml_native_vulkan_devices, AN_DIRTY},
  {"upload", "(Int, List[Int], List[Float], Bool) -> Result[TensorHandle]", ggml_native_upload, AN_DIRTY},
  {"load", "(Int, String, Int, List[Int], Int, Int) -> Result[TensorHandle]", ggml_native_load, AN_DIRTY},
  {"from_bytes", "(Int, Bytes, List[Int], Int, Int) -> Result[TensorHandle]", ggml_native_from_bytes, AN_DIRTY},
  {"filled", "(Int, List[Int], Float, Bool) -> Result[TensorHandle]", ggml_native_filled, AN_DIRTY},
  {"on_cpu", "(TensorHandle) -> Bool", ggml_native_on_cpu, AN_SHORT},
  {"readback", "(TensorHandle) -> Result[List[Float]]", ggml_native_readback, AN_DIRTY},
  {"append_file", "(TensorHandle, String) -> Result[Void]", ggml_native_append_file, AN_DIRTY},
  {"transfer", "(TensorHandle, Int) -> Result[TensorHandle]", ggml_native_transfer, AN_DIRTY},
  {"acquire", "(Int, String, Int, Bool) -> Result[Lease]", ggml_native_acquire, AN_DIRTY},
  {"ready", "(Lease) -> Bool", ggml_native_ready, AN_SHORT},
  {"node", "(Lease, Int, List[Int], List[Int], List[Int]) -> Result[Int]", ggml_native_node, AN_DIRTY},
  {"compile", "(Lease, List[Int]) -> Result[Void]", ggml_native_compile, AN_DIRTY},
  {"bind", "(Lease, Int, TensorHandle) -> Result[Void]", ggml_native_bind, AN_SHORT},
  {"output", "(Lease, Int, Bool) -> Result[TensorHandle]", ggml_native_output, AN_DIRTY},
  {"output_into", "(Lease, Int, TensorHandle) -> Result[Void]", ggml_native_output_into, AN_DIRTY},
  {"pin", "(Lease) -> Result[Pin]", ggml_native_pin, AN_DIRTY},
  {"pin_id", "(Pin) -> Int", ggml_native_pin_id, AN_SHORT},
  {"resume", "(Lease, Int) -> Bool", ggml_native_resume, AN_SHORT},
  {"keep", "(Lease, Int, List[Int]) -> Void", ggml_native_keep, AN_SHORT},
  {"kept", "(Lease) -> Int", ggml_native_kept, AN_SHORT},
  {"run", "(Lease, List[Float]) -> Result[Void]", ggml_native_run, AN_DIRTY},
  {"convert_file", "(String, Int, Int, Int, String, Int, Int) -> Result[Void]", ggml_native_convert_file, AN_DIRTY},
  {"compiles", "() -> Int", ggml_native_compiles, AN_DIRTY},
  {"cpu_threads", "() -> Result[Int]", ggml_native_cpu_threads, AN_DIRTY},
  {"vulkan_fp16", "(Bool) -> Result[Void]", ggml_native_vulkan_fp16, AN_DIRTY},
};

/* Ends the module's one .c file: the tables and agentica_native_init. LOAD is NULL or a const char *(*)(void)
   that runs once at the first call, and refuses the module with the message it returns. */
#define GGML_NATIVE_MODULE(LOAD) \
  static const an_resource_type ggml_native_resources[] = { \
    {"TensorHandle", sizeof(struct ggml_TensorHandle_state), GGML_TensorHandle_FREE}, \
    {"Lease", sizeof(struct ggml_Lease_state), GGML_Lease_FREE}, \
    {"Pin", sizeof(struct ggml_Pin_state), GGML_Pin_FREE}, \
  }; \
  AN_MODULE_TABLES("ggml", ggml_native_functions, 26, NULL, 0, ggml_native_resources, 3, LOAD)
#endif
