/* The C side of `extern module "ggmlgraph"` in ggml_graph.agc, made by native-gen.
   Do not edit it: change the .agc and make it again. */
#ifndef GGMLGRAPH_NATIVE_H
#define GGMLGRAPH_NATIVE_H

#include "agentica_native.h"

#if AN_API_MAJOR != 1
#error "ggmlgraph_native.h needs version 1 of the native API (agentica_native.h)"
#endif
#if AN_API_MINOR < 2
#error "ggmlgraph_native.h needs version 1.2 or later of the native API (agentica_native.h)"
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
enum { GGMLGRAPH_RESOURCE_Backend = 0, GGMLGRAPH_RESOURCE_Program = 1 };

/* An open backend: ggml's CPU backend with a thread count, or the iGPU
   through ggml's Vulkan backend. The state owns the `ggml_backend_t`.

   Define it, and GGMLGRAPH_Backend_FREE (a void (*)(void *state), or NULL), before GGMLGRAPH_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct ggmlgraph_Backend_state;
static inline struct ggmlgraph_Backend_state *ggmlgraph_Backend_new(an_env *env, an_term *out) {
  return (struct ggmlgraph_Backend_state *)an_resource_new(env, GGMLGRAPH_RESOURCE_Backend, out);
}
static inline struct ggmlgraph_Backend_state *ggmlgraph_Backend_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, GGMLGRAPH_RESOURCE_Backend, &state) ? (struct ggmlgraph_Backend_state *)state : NULL;
}

/* One compiled training step: its ggml contexts, graph, buffers and the
   host batches of a benchmark. It borrows the `ggml_backend_t` of the
   `Backend` it was made on, which must stay open while it is used.

   Define it, and GGMLGRAPH_Program_FREE (a void (*)(void *state), or NULL), before GGMLGRAPH_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct ggmlgraph_Program_state;
static inline struct ggmlgraph_Program_state *ggmlgraph_Program_new(an_env *env, an_term *out) {
  return (struct ggmlgraph_Program_state *)an_resource_new(env, GGMLGRAPH_RESOURCE_Program, out);
}
static inline struct ggmlgraph_Program_state *ggmlgraph_Program_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, GGMLGRAPH_RESOURCE_Program, &state) ? (struct ggmlgraph_Program_state *)state : NULL;
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
} ggmlgraph_list_Int;
bool ggmlgraph_list_Int_open(an_env *env, an_term t, ggmlgraph_list_Int *out);
int ggmlgraph_list_Int_next(an_env *env, ggmlgraph_list_Int *list, int64_t *out);

typedef struct {
  an_term term;
  an_list_cursor cursor;
  size_t len;
  size_t at;
  bool failed;
  int argument; /* 1 + the index of the argument of the call it was opened for, or 0 (_open) */
} ggmlgraph_list_Float;
bool ggmlgraph_list_Float_open(an_env *env, an_term t, ggmlgraph_list_Float *out);
int ggmlgraph_list_Float_next(an_env *env, ggmlgraph_list_Float *list, double *out);

an_term ggmlgraph_result_Backend_ok(an_env *env, an_term value);

an_term ggmlgraph_result_String_ok(an_env *env, ng_str value);

an_term ggmlgraph_result_Program_ok(an_env *env, an_term value);

an_term ggmlgraph_result_Int_ok(an_env *env, int64_t value);

an_term ggmlgraph_result_Void_ok(an_env *env, an_term value);

an_term ggmlgraph_result_Float_ok(an_env *env, double value);

typedef struct {
  an_list_builder builder;
} ggmlgraph_list_Float_builder;
void ggmlgraph_list_Float_begin(an_env *env, ggmlgraph_list_Float_builder *b, size_t hint);
void ggmlgraph_list_Float_push(an_env *env, ggmlgraph_list_Float_builder *b, double value);
an_term ggmlgraph_list_Float_end(an_env *env, ggmlgraph_list_Float_builder *b);

an_term ggmlgraph_result_list_Float_ok(an_env *env, an_term value);

typedef struct {
  an_list_builder builder;
} ggmlgraph_list_Int_builder;
void ggmlgraph_list_Int_begin(an_env *env, ggmlgraph_list_Int_builder *b, size_t hint);
void ggmlgraph_list_Int_push(an_env *env, ggmlgraph_list_Int_builder *b, int64_t value);
an_term ggmlgraph_list_Int_end(an_env *env, ggmlgraph_list_Int_builder *b);

an_term ggmlgraph_result_list_Int_ok(an_env *env, an_term value);

/* The CPU backend with `threads` threads. */
an_term ggmlgraph_backend_cpu(an_env *env, int64_t threads);

/* Vulkan device 0, failing unless it is Vulkan and not a CPU. */
an_term ggmlgraph_backend_vulkan(an_env *env);

/* `"name — description (type)"` of the backend's device. */
an_term ggmlgraph_describe(an_env *env, struct ggmlgraph_Backend_state *backend);

/* An empty program with room for `capacity` nodes. */
an_term ggmlgraph_program(an_env *env, struct ggmlgraph_Backend_state *backend, int64_t capacity);

/* Appends one node and answers its index: `op` is a code of
   `native/ggmlgraph.c`, `args` earlier indices, `shape` the row-major shape
   the trace recorded, `ints` the op's integer parameters (a leaf's slot, a
   reduction's axes, arange's start, stop and step) and `value` a fill's
   value. */
an_term ggmlgraph_node(an_env *env, struct ggmlgraph_Program_state *p, int64_t op, ggmlgraph_list_Int args, ggmlgraph_list_Int shape, ggmlgraph_list_Int ints, double value);

/* Points the leaf `node` of `p` at the leaf `source` of `other`, so two
   programs share one buffer (the two-graph optimizer). */
an_term ggmlgraph_share(an_env *env, struct ggmlgraph_Program_state *p, int64_t node, struct ggmlgraph_Program_state *other, int64_t source);

/* Builds the graph of `outputs` and of the writebacks (pairs of a state
   leaf and the node whose value replaces it after each run), allocates it
   and checks every node against the backend. */
an_term ggmlgraph_compile(an_env *env, struct ggmlgraph_Program_state *p, ggmlgraph_list_Int outputs, ggmlgraph_list_Int writebacks);

/* Uploads row-major `values` into the leaf `node`. */
an_term ggmlgraph_set(an_env *env, struct ggmlgraph_Program_state *p, int64_t node, ggmlgraph_list_Float values);

/* `count` host batches of synthetic data for the leaves `x` (`[b, d]`
   floats) and `y` (`[b]` labels below `classes`), made in C from `seed`. */
an_term ggmlgraph_batches(an_env *env, struct ggmlgraph_Program_state *p, int64_t x, int64_t y, int64_t classes, int64_t count, int64_t seed);

/* One step: uploads host batch `batch` (when `batches` made any), runs
   the graph, waits, and answers the first output's first value (the loss). */
an_term ggmlgraph_run(an_env *env, struct ggmlgraph_Program_state *p, int64_t batch);

/* The values of node `node` after the last run. */
an_term ggmlgraph_read(an_env *env, struct ggmlgraph_Program_state *p, int64_t node);

/* What the program costs: graph nodes, leaf bytes, compute buffer bytes,
   compile ns, runs, upload ns, compute ns (totals over the runs). */
an_term ggmlgraph_stats(an_env *env, struct ggmlgraph_Program_state *p);

#ifdef __cplusplus
}
#endif

#endif

/* Outside the include guard, so a file that included the header before it defined the switch still gets this
   part; its own guard keeps it to one copy. */
#if defined(GGMLGRAPH_NATIVE_IMPLEMENTATION) && !defined(GGMLGRAPH_NATIVE_IMPLEMENTATION_DONE)
#define GGMLGRAPH_NATIVE_IMPLEMENTATION_DONE
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

static bool ggmlgraph_native_item_Int(an_env *env, an_term x, int64_t *out, ng_fail *fail) {
  return an_get_int(env, x, &(*out)) || ng_expected(fail, "Int");
}

static bool ggmlgraph_native_open_list_Int(an_env *env, an_term t, ggmlgraph_list_Int *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Int]");
}

bool ggmlgraph_list_Int_open(an_env *env, an_term t, ggmlgraph_list_Int *out) {
  ng_fail fail;
  return ggmlgraph_native_open_list_Int(env, t, out, &fail) || ng_raise(env, "", "List[Int]", &fail);
}

int ggmlgraph_list_Int_next(an_env *env, ggmlgraph_list_Int *list, int64_t *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (ggmlgraph_native_item_Int(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Int]", list->argument, at, NULL);
}

static bool ggmlgraph_native_item_Float(an_env *env, an_term x, double *out, ng_fail *fail) {
  return an_get_float(env, x, &(*out)) || ng_expected(fail, "Float");
}

static bool ggmlgraph_native_open_list_Float(an_env *env, an_term t, ggmlgraph_list_Float *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Float]");
}

bool ggmlgraph_list_Float_open(an_env *env, an_term t, ggmlgraph_list_Float *out) {
  ng_fail fail;
  return ggmlgraph_native_open_list_Float(env, t, out, &fail) || ng_raise(env, "", "List[Float]", &fail);
}

int ggmlgraph_list_Float_next(an_env *env, ggmlgraph_list_Float *list, double *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (ggmlgraph_native_item_Float(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Float]", list->argument, at, NULL);
}

an_term ggmlgraph_result_Backend_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggmlgraph_result_String_ok(an_env *env, ng_str value) {
  return an_ok(env, an_string(env, value.data, value.len));
}

an_term ggmlgraph_result_Program_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggmlgraph_result_Int_ok(an_env *env, int64_t value) {
  return an_ok(env, an_int(env, value));
}

an_term ggmlgraph_result_Void_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term ggmlgraph_result_Float_ok(an_env *env, double value) {
  return an_ok(env, an_float(env, value));
}

void ggmlgraph_list_Float_begin(an_env *env, ggmlgraph_list_Float_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void ggmlgraph_list_Float_push(an_env *env, ggmlgraph_list_Float_builder *b, double value) {
  an_list_push(env, &b->builder, an_float(env, value));
}

an_term ggmlgraph_list_Float_end(an_env *env, ggmlgraph_list_Float_builder *b) { return an_list_end(env, &b->builder); }

an_term ggmlgraph_result_list_Float_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

void ggmlgraph_list_Int_begin(an_env *env, ggmlgraph_list_Int_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void ggmlgraph_list_Int_push(an_env *env, ggmlgraph_list_Int_builder *b, int64_t value) {
  an_list_push(env, &b->builder, an_int(env, value));
}

an_term ggmlgraph_list_Int_end(an_env *env, ggmlgraph_list_Int_builder *b) { return an_list_end(env, &b->builder); }

an_term ggmlgraph_result_list_Int_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

static an_term ggmlgraph_native_backend_cpu(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  int64_t a0;
  if (!an_get_int(env, argv[0], &a0)) return an_badarg(env, 0);
  return ggmlgraph_backend_cpu(env, a0);
}

static an_term ggmlgraph_native_backend_vulkan(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  (void)argv;
  return ggmlgraph_backend_vulkan(env);
}

static an_term ggmlgraph_native_describe(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlgraph_Backend_state *a0 = ggmlgraph_Backend_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggmlgraph_describe(env, a0);
}

static an_term ggmlgraph_native_program(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlgraph_Backend_state *a0 = ggmlgraph_Backend_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  return ggmlgraph_program(env, a0, a1);
}

static an_term ggmlgraph_native_node(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct ggmlgraph_Program_state *a0 = ggmlgraph_Program_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  ggmlgraph_list_Int a2;
  if (!ggmlgraph_native_open_list_Int(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 2);
  a2.argument = 3;
  ggmlgraph_list_Int a3;
  if (!ggmlgraph_native_open_list_Int(env, argv[3], &a3, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 3);
  a3.argument = 4;
  ggmlgraph_list_Int a4;
  if (!ggmlgraph_native_open_list_Int(env, argv[4], &a4, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 4);
  a4.argument = 5;
  double a5;
  if (!an_get_float(env, argv[5], &a5)) return an_badarg(env, 5);
  return ggmlgraph_node(env, a0, a1, a2, a3, a4, a5);
}

static an_term ggmlgraph_native_share(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlgraph_Program_state *a0 = ggmlgraph_Program_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  struct ggmlgraph_Program_state *a2 = ggmlgraph_Program_get(env, argv[2]);
  if (!a2) return an_badarg(env, 2);
  int64_t a3;
  if (!an_get_int(env, argv[3], &a3)) return an_badarg(env, 3);
  return ggmlgraph_share(env, a0, a1, a2, a3);
}

static an_term ggmlgraph_native_compile(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct ggmlgraph_Program_state *a0 = ggmlgraph_Program_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  ggmlgraph_list_Int a1;
  if (!ggmlgraph_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  ggmlgraph_list_Int a2;
  if (!ggmlgraph_native_open_list_Int(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 2);
  a2.argument = 3;
  return ggmlgraph_compile(env, a0, a1, a2);
}

static an_term ggmlgraph_native_set(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct ggmlgraph_Program_state *a0 = ggmlgraph_Program_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  ggmlgraph_list_Float a2;
  if (!ggmlgraph_native_open_list_Float(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Float]", 2);
  a2.argument = 3;
  return ggmlgraph_set(env, a0, a1, a2);
}

static an_term ggmlgraph_native_batches(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlgraph_Program_state *a0 = ggmlgraph_Program_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  int64_t a3;
  if (!an_get_int(env, argv[3], &a3)) return an_badarg(env, 3);
  int64_t a4;
  if (!an_get_int(env, argv[4], &a4)) return an_badarg(env, 4);
  int64_t a5;
  if (!an_get_int(env, argv[5], &a5)) return an_badarg(env, 5);
  return ggmlgraph_batches(env, a0, a1, a2, a3, a4, a5);
}

static an_term ggmlgraph_native_run(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlgraph_Program_state *a0 = ggmlgraph_Program_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  return ggmlgraph_run(env, a0, a1);
}

static an_term ggmlgraph_native_read(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlgraph_Program_state *a0 = ggmlgraph_Program_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  return ggmlgraph_read(env, a0, a1);
}

static an_term ggmlgraph_native_stats(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct ggmlgraph_Program_state *a0 = ggmlgraph_Program_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return ggmlgraph_stats(env, a0);
}

static const an_function ggmlgraph_native_functions[] = {
  {"backend_cpu", "(Int) -> Result[Backend]", ggmlgraph_native_backend_cpu, AN_DIRTY},
  {"backend_vulkan", "() -> Result[Backend]", ggmlgraph_native_backend_vulkan, AN_DIRTY},
  {"describe", "(Backend) -> Result[String]", ggmlgraph_native_describe, AN_DIRTY},
  {"program", "(Backend, Int) -> Result[Program]", ggmlgraph_native_program, AN_DIRTY},
  {"node", "(Program, Int, List[Int], List[Int], List[Int], Float) -> Result[Int]", ggmlgraph_native_node, AN_DIRTY},
  {"share", "(Program, Int, Program, Int) -> Result[Void]", ggmlgraph_native_share, AN_DIRTY},
  {"compile", "(Program, List[Int], List[Int]) -> Result[Void]", ggmlgraph_native_compile, AN_DIRTY},
  {"set", "(Program, Int, List[Float]) -> Result[Void]", ggmlgraph_native_set, AN_DIRTY},
  {"batches", "(Program, Int, Int, Int, Int, Int) -> Result[Void]", ggmlgraph_native_batches, AN_DIRTY},
  {"run", "(Program, Int) -> Result[Float]", ggmlgraph_native_run, AN_DIRTY},
  {"read", "(Program, Int) -> Result[List[Float]]", ggmlgraph_native_read, AN_DIRTY},
  {"stats", "(Program) -> Result[List[Int]]", ggmlgraph_native_stats, AN_DIRTY},
};

/* Ends the module's one .c file: the tables and agentica_native_init. LOAD is NULL or a const char *(*)(void)
   that runs once at the first call, and refuses the module with the message it returns. */
#define GGMLGRAPH_NATIVE_MODULE(LOAD) \
  static const an_resource_type ggmlgraph_native_resources[] = { \
    {"Backend", sizeof(struct ggmlgraph_Backend_state), GGMLGRAPH_Backend_FREE}, \
    {"Program", sizeof(struct ggmlgraph_Program_state), GGMLGRAPH_Program_FREE}, \
  }; \
  AN_MODULE_TABLES("ggmlgraph", ggmlgraph_native_functions, 12, NULL, 0, ggmlgraph_native_resources, 2, LOAD)
#endif
