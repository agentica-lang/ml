/* The C side of `extern module "ggmlspike"` in ggml_spike.agc, made by native-gen.
   Do not edit it: change the .agc and make it again. */
#ifndef GGMLSPIKE_NATIVE_H
#define GGMLSPIKE_NATIVE_H

#include "agentica_native.h"

#if AN_API_MAJOR != 1
#error "ggmlspike_native.h needs version 1 of the native API (agentica_native.h)"
#endif
#if AN_API_MINOR < 2
#error "ggmlspike_native.h needs version 1.2 or later of the native API (agentica_native.h)"
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
enum { GGMLSPIKE_RESOURCE_Backend = 0, GGMLSPIKE_RESOURCE_Tensor = 1 };

/* An open backend: ggml's CPU backend with a thread count, or the iGPU
   through ggml's Vulkan backend. The state owns the `ggml_backend_t`.

   Define it, and GGMLSPIKE_Backend_FREE (a void (*)(void *state), or NULL), before GGMLSPIKE_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct ggmlspike_Backend_state;
static inline struct ggmlspike_Backend_state *ggmlspike_Backend_new(an_env *env, an_term *out) {
  return (struct ggmlspike_Backend_state *)an_resource_new(env, GGMLSPIKE_RESOURCE_Backend, out);
}
static inline struct ggmlspike_Backend_state *ggmlspike_Backend_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, GGMLSPIKE_RESOURCE_Backend, &state) ? (struct ggmlspike_Backend_state *)state : NULL;
}

/* A F32 tensor whose state owns its ggml context and backend buffer. The
   `Backend` it was made on must outlive it (the spike keeps backends open
   for the whole run).

   Define it, and GGMLSPIKE_Tensor_FREE (a void (*)(void *state), or NULL), before GGMLSPIKE_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct ggmlspike_Tensor_state;
static inline struct ggmlspike_Tensor_state *ggmlspike_Tensor_new(an_env *env, an_term *out) {
  return (struct ggmlspike_Tensor_state *)an_resource_new(env, GGMLSPIKE_RESOURCE_Tensor, out);
}
static inline struct ggmlspike_Tensor_state *ggmlspike_Tensor_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, GGMLSPIKE_RESOURCE_Tensor, &state) ? (struct ggmlspike_Tensor_state *)state : NULL;
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
} ggmlspike_list_Int;
bool ggmlspike_list_Int_open(an_env *env, an_term t, ggmlspike_list_Int *out);
int ggmlspike_list_Int_next(an_env *env, ggmlspike_list_Int *list, int64_t *out);

typedef struct {
  an_list_builder builder;
} ggmlspike_list_String_builder;
void ggmlspike_list_String_begin(an_env *env, ggmlspike_list_String_builder *b, size_t hint);
void ggmlspike_list_String_push(an_env *env, ggmlspike_list_String_builder *b, ng_str value);
an_term ggmlspike_list_String_end(an_env *env, ggmlspike_list_String_builder *b);

an_term ggmlspike_result_list_String_ok(an_env *env, an_term value);

an_term ggmlspike_result_Backend_ok(an_env *env, an_term value);

an_term ggmlspike_result_String_ok(an_env *env, ng_str value);

an_term ggmlspike_result_Tensor_ok(an_env *env, an_term value);

an_term ggmlspike_result_Void_ok(an_env *env, an_term value);

an_term ggmlspike_result_Float_ok(an_env *env, double value);

typedef struct {
  an_list_builder builder;
} ggmlspike_list_Float_builder;
void ggmlspike_list_Float_begin(an_env *env, ggmlspike_list_Float_builder *b, size_t hint);
void ggmlspike_list_Float_push(an_env *env, ggmlspike_list_Float_builder *b, double value);
an_term ggmlspike_list_Float_end(an_env *env, ggmlspike_list_Float_builder *b);

an_term ggmlspike_result_list_Float_ok(an_env *env, an_term value);

an_term ggmlspike_result_Int_ok(an_env *env, int64_t value);

typedef struct {
  an_list_builder builder;
} ggmlspike_list_Int_builder;
void ggmlspike_list_Int_begin(an_env *env, ggmlspike_list_Int_builder *b, size_t hint);
void ggmlspike_list_Int_push(an_env *env, ggmlspike_list_Int_builder *b, int64_t value);
an_term ggmlspike_list_Int_end(an_env *env, ggmlspike_list_Int_builder *b);

an_term ggmlspike_result_list_Int_ok(an_env *env, an_term value);

/* Every device ggml currently registers, one `"name — description (type)"`
   per entry, in ggml's order. Reading it opens nothing. */
an_term ggmlspike_devices(an_env *env);

/* ggml's CPU backend, with `threads` threads for both the graph and batch
   paths. */
an_term ggmlspike_backend_cpu(an_env *env, int64_t threads);

/* The iGPU through ggml's Vulkan backend (`ggml_backend_vk_init(0)`),
   failing unless the backend really is Vulkan and the device is not a CPU. */
an_term ggmlspike_backend_vulkan(an_env *env);

/* What the backend really is, read back from ggml: name, device
   description and device type. The type is the assertion of this spike. */
an_term ggmlspike_describe(an_env *env, struct ggmlspike_Backend_state *backend);

/* A `shape` tensor of F32 filled with the deterministic pattern of
   `tensor/tools/bench.agc`, so every backend gets the same values. */
an_term ggmlspike_create(an_env *env, struct ggmlspike_Backend_state *backend, ggmlspike_list_Int shape);

/* Elementwise `a + b`, both of one shape; error when they differ. */
an_term ggmlspike_add(an_env *env, struct ggmlspike_Tensor_state *a, struct ggmlspike_Tensor_state *b);

/* `max(0, x)`, elementwise. */
an_term ggmlspike_relu(an_env *env, struct ggmlspike_Tensor_state *a);

/* ggml's `mul_mat(a, b)`, which is `a · bᵀ`: for `a` of `[k, m]` and `b` of
   `[k, n]` (row-major rows), the result is `[m, n]`. */
an_term ggmlspike_mul_mat(an_env *env, struct ggmlspike_Tensor_state *a, struct ggmlspike_Tensor_state *b);

/* One dispatch round trip with no op in it: an empty graph computed on the
   backend. This is the fixed cost every op above pays before it does work. */
an_term ggmlspike_empty_graph(an_env *env, struct ggmlspike_Backend_state *backend);

/* The sum of every element, read back across the boundary as one `Float`:
   a full device-to-host copy plus a host sum, with no `List` decode. */
an_term ggmlspike_checksum(an_env *env, struct ggmlspike_Tensor_state *t);

/* Every element, read back as a `List[Float]` (the slow path M0 measured at
   ~20 ns per value; `checksum` is the cost of the copy alone). */
an_term ggmlspike_readback(an_env *env, struct ggmlspike_Tensor_state *t);

/* The bytes the backend buffer holding `t` actually has, as ggml reports
   them. The only memory number this spike can report honestly. */
an_term ggmlspike_allocated(an_env *env, struct ggmlspike_Tensor_state *t);

/* The device's free and total memory in bytes, as ggml reports them: on a
   Vega iGPU both are host memory. */
an_term ggmlspike_memory(an_env *env, struct ggmlspike_Backend_state *backend);

#ifdef __cplusplus
}
#endif

#endif

/* Outside the include guard, so a file that included the header before it defined the switch still gets this
   part; its own guard keeps it to one copy. */
#if defined(GGMLSPIKE_NATIVE_IMPLEMENTATION) && !defined(GGMLSPIKE_NATIVE_IMPLEMENTATION_DONE)
#define GGMLSPIKE_NATIVE_IMPLEMENTATION_DONE
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

static bool ggmlspike_native_item_Int(an_env *env, an_term x, int64_t *out, ng_fail *fail) {
  return an_get_int(env, x, &(*out)) || ng_expected(fail, "Int");
}

static bool ggmlspike_native_open_list_Int(an_env *env, an_term t, ggmlspike_list_Int *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Int]");
}

bool ggmlspike_list_Int_open(an_env *env, an_term t, ggmlspike_list_Int *out) {
  ng_fail fail;
  return ggmlspike_native_open_list_Int(env, t, out, &fail) || ng_raise(env, "", "List[Int]", &fail);
}

int ggmlspike_list_Int_next(an_env *env, ggmlspike_list_Int *list, int64_t *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (ggmlspike_native_item_Int(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Int]", list->argument, at, NULL);
}

void ggmlspike_list_String_begin(an_env *env, ggmlspike_list_String_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void ggmlspike_list_String_push(an_env *env, ggmlspike_list_String_builder *b, ng_str value) {
  an_list_push(env, &b->builder, an_string(env, value.data, value.len));
}

an_term ggmlspike_list_String_end(an_env *env, ggmlspike_list_String_builder *b) { return an_list_end(env, &b->builder); }

an_term ggmlspike_result_list_String_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggmlspike_result_Backend_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggmlspike_result_String_ok(an_env *env, ng_str value) {
  return an_ok(env, an_string(env, value.data, value.len));
}

an_term ggmlspike_result_Tensor_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggmlspike_result_Void_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggmlspike_result_Float_ok(an_env *env, double value) {
  return an_ok(env, an_float(env, value));
}

void ggmlspike_list_Float_begin(an_env *env, ggmlspike_list_Float_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void ggmlspike_list_Float_push(an_env *env, ggmlspike_list_Float_builder *b, double value) {
  an_list_push(env, &b->builder, an_float(env, value));
}

an_term ggmlspike_list_Float_end(an_env *env, ggmlspike_list_Float_builder *b) { return an_list_end(env, &b->builder); }

an_term ggmlspike_result_list_Float_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggmlspike_result_Int_ok(an_env *env, int64_t value) {
  return an_ok(env, an_int(env, value));
}

void ggmlspike_list_Int_begin(an_env *env, ggmlspike_list_Int_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void ggmlspike_list_Int_push(an_env *env, ggmlspike_list_Int_builder *b, int64_t value) {
  an_list_push(env, &b->builder, an_int(env, value));
}

an_term ggmlspike_list_Int_end(an_env *env, ggmlspike_list_Int_builder *b) { return an_list_end(env, &b->builder); }

an_term ggmlspike_result_list_Int_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

static an_term ggmlspike_native_devices(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  (void)argv;
  return ggmlspike_devices(env);
}

static an_term ggmlspike_native_backend_cpu(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  int64_t a0;
  if (!an_get_int(env, argv[0], &a0)) return an_badarg(env, 0);
  return ggmlspike_backend_cpu(env, a0);
}

static an_term ggmlspike_native_backend_vulkan(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  (void)argv;
  return ggmlspike_backend_vulkan(env);
}

static an_term ggmlspike_native_describe(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlspike_Backend_state *a0 = ggmlspike_Backend_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggmlspike_describe(env, a0);
}

static an_term ggmlspike_native_create(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct ggmlspike_Backend_state *a0 = ggmlspike_Backend_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  ggmlspike_list_Int a1;
  if (!ggmlspike_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  return ggmlspike_create(env, a0, a1);
}

static an_term ggmlspike_native_add(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlspike_Tensor_state *a0 = ggmlspike_Tensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct ggmlspike_Tensor_state *a1 = ggmlspike_Tensor_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return ggmlspike_add(env, a0, a1);
}

static an_term ggmlspike_native_relu(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlspike_Tensor_state *a0 = ggmlspike_Tensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggmlspike_relu(env, a0);
}

static an_term ggmlspike_native_mul_mat(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlspike_Tensor_state *a0 = ggmlspike_Tensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct ggmlspike_Tensor_state *a1 = ggmlspike_Tensor_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return ggmlspike_mul_mat(env, a0, a1);
}

static an_term ggmlspike_native_empty_graph(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlspike_Backend_state *a0 = ggmlspike_Backend_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggmlspike_empty_graph(env, a0);
}

static an_term ggmlspike_native_checksum(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlspike_Tensor_state *a0 = ggmlspike_Tensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggmlspike_checksum(env, a0);
}

static an_term ggmlspike_native_readback(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlspike_Tensor_state *a0 = ggmlspike_Tensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggmlspike_readback(env, a0);
}

static an_term ggmlspike_native_allocated(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlspike_Tensor_state *a0 = ggmlspike_Tensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggmlspike_allocated(env, a0);
}

static an_term ggmlspike_native_memory(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlspike_Backend_state *a0 = ggmlspike_Backend_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggmlspike_memory(env, a0);
}

static const an_function ggmlspike_native_functions[] = {
  {"devices", "() -> Result[List[String]]", ggmlspike_native_devices, AN_DIRTY},
  {"backend_cpu", "(Int) -> Result[Backend]", ggmlspike_native_backend_cpu, AN_DIRTY},
  {"backend_vulkan", "() -> Result[Backend]", ggmlspike_native_backend_vulkan, AN_DIRTY},
  {"describe", "(Backend) -> Result[String]", ggmlspike_native_describe, AN_DIRTY},
  {"create", "(Backend, List[Int]) -> Result[Tensor]", ggmlspike_native_create, AN_DIRTY},
  {"add", "(Tensor, Tensor) -> Result[Tensor]", ggmlspike_native_add, AN_DIRTY},
  {"relu", "(Tensor) -> Result[Tensor]", ggmlspike_native_relu, AN_DIRTY},
  {"mul_mat", "(Tensor, Tensor) -> Result[Tensor]", ggmlspike_native_mul_mat, AN_DIRTY},
  {"empty_graph", "(Backend) -> Result[Void]", ggmlspike_native_empty_graph, AN_DIRTY},
  {"checksum", "(Tensor) -> Result[Float]", ggmlspike_native_checksum, AN_DIRTY},
  {"readback", "(Tensor) -> Result[List[Float]]", ggmlspike_native_readback, AN_DIRTY},
  {"allocated", "(Tensor) -> Result[Int]", ggmlspike_native_allocated, AN_DIRTY},
  {"memory", "(Backend) -> Result[List[Int]]", ggmlspike_native_memory, AN_DIRTY},
};

/* Ends the module's one .c file: the tables and agentica_native_init. LOAD is NULL or a const char *(*)(void)
   that runs once at the first call, and refuses the module with the message it returns. */
#define GGMLSPIKE_NATIVE_MODULE(LOAD) \
  static const an_resource_type ggmlspike_native_resources[] = { \
    {"Backend", sizeof(struct ggmlspike_Backend_state), GGMLSPIKE_Backend_FREE}, \
    {"Tensor", sizeof(struct ggmlspike_Tensor_state), GGMLSPIKE_Tensor_FREE}, \
  }; \
  AN_MODULE_TABLES("ggmlspike", ggmlspike_native_functions, 13, NULL, 0, ggmlspike_native_resources, 2, LOAD)
#endif
