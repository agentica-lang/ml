/* The C side of `extern module "xla"` in xla_native.agc, made by native-gen.
   Do not edit it: change the .agc and make it again. */
#ifndef XLA_NATIVE_H
#define XLA_NATIVE_H

#include "agentica_native.h"

#if AN_API_MAJOR != 1
#error "xla_native.h needs version 1 of the native API (agentica_native.h)"
#endif
#if AN_API_MINOR < 2
#error "xla_native.h needs version 1.2 or later of the native API (agentica_native.h)"
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
enum { XLA_RESOURCE_Buffer = 0, XLA_RESOURCE_Program = 1, XLA_RESOURCE_Call = 2 };

/* One PJRT buffer on the CPU device of the plugin, destroyed by the free hook.

   Define it, and XLA_Buffer_FREE (a void (*)(void *state), or NULL), before XLA_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct xla_Buffer_state;
static inline struct xla_Buffer_state *xla_Buffer_new(an_env *env, an_term *out) {
  return (struct xla_Buffer_state *)an_resource_new(env, XLA_RESOURCE_Buffer, out);
}
static inline struct xla_Buffer_state *xla_Buffer_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, XLA_RESOURCE_Buffer, &state) ? (struct xla_Buffer_state *)state : NULL;
}

/* One program compiled by the plugin (a PJRT loaded executable), destroyed by the free hook.

   Define it, and XLA_Program_FREE (a void (*)(void *state), or NULL), before XLA_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct xla_Program_state;
static inline struct xla_Program_state *xla_Program_new(an_env *env, an_term *out) {
  return (struct xla_Program_state *)an_resource_new(env, XLA_RESOURCE_Program, out);
}
static inline struct xla_Program_state *xla_Program_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, XLA_RESOURCE_Program, &state) ? (struct xla_Program_state *)state : NULL;
}

/* The arguments of one run of a program, collected by `arg`: it holds each buffer until it is freed, so a buffer
   closed meanwhile is still there for the run.

   Define it, and XLA_Call_FREE (a void (*)(void *state), or NULL), before XLA_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct xla_Call_state;
static inline struct xla_Call_state *xla_Call_new(an_env *env, an_term *out) {
  return (struct xla_Call_state *)an_resource_new(env, XLA_RESOURCE_Call, out);
}
static inline struct xla_Call_state *xla_Call_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, XLA_RESOURCE_Call, &state) ? (struct xla_Call_state *)state : NULL;
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
} xla_list_Int;
bool xla_list_Int_open(an_env *env, an_term t, xla_list_Int *out);
int xla_list_Int_next(an_env *env, xla_list_Int *list, int64_t *out);

typedef struct {
  an_term term;
  an_list_cursor cursor;
  size_t len;
  size_t at;
  bool failed;
  int argument; /* 1 + the index of the argument of the call it was opened for, or 0 (_open) */
} xla_list_Float;
bool xla_list_Float_open(an_env *env, an_term t, xla_list_Float *out);
int xla_list_Float_next(an_env *env, xla_list_Float *list, double *out);

an_term xla_result_Void_ok(an_env *env, an_term value);

an_term xla_result_Buffer_ok(an_env *env, an_term value);

typedef struct {
  an_list_builder builder;
} xla_list_Float_builder;
void xla_list_Float_begin(an_env *env, xla_list_Float_builder *b, size_t hint);
void xla_list_Float_push(an_env *env, xla_list_Float_builder *b, double value);
an_term xla_list_Float_end(an_env *env, xla_list_Float_builder *b);

an_term xla_result_list_Float_ok(an_env *env, an_term value);

an_term xla_result_Program_ok(an_env *env, an_term value);

typedef struct {
  an_list_builder builder;
} xla_list_Buffer_builder;
void xla_list_Buffer_begin(an_env *env, xla_list_Buffer_builder *b, size_t hint);
void xla_list_Buffer_push(an_env *env, xla_list_Buffer_builder *b, an_term value);
an_term xla_list_Buffer_end(an_env *env, xla_list_Buffer_builder *b);

an_term xla_result_list_Buffer_ok(an_env *env, an_term value);

/* The machine this module was built for: `x86_64`, `aarch64` or `other`. It loads no plugin. */
an_term xla_arch(an_env *env);

/* Loads the plugin at `path` and makes its client, once for the process; a later call with the same path does
   nothing, and with another path is an error. A failure leaves no client, so a later call may try again. On
   x86_64 a CPU without AVX is an error, before anything loads. */
an_term xla_open(an_env *env, ng_str path);

/* A buffer of the row-major `shape` and the dtype `dtype` holding `data`, one value per element. */
an_term xla_upload(an_env *env, xla_list_Int shape, xla_list_Float data, int64_t dtype);

/* An `F32` buffer of `shape` from the range at `offset` of the file `path`, which `format` (a `Types.file_code`)
   holds: the call reads and widens it in C, and checks the range against the file. */
an_term xla_load(an_env *env, ng_str path, int64_t offset, xla_list_Int shape, int64_t format);

/* The same from `data`, the bytes of the range that Agentica read: they must be exactly the range. */
an_term xla_from_bytes(an_env *env, ng_buf data, xla_list_Int shape, int64_t format);

/* A buffer of `shape` and `dtype` with every element `value`, filled in C. */
an_term xla_filled(an_env *env, xla_list_Int shape, double value, int64_t dtype);

/* Host readback: the row-major values as doubles, which the caller tags with the tensor's dtype. */
an_term xla_readback(an_env *env, struct xla_Buffer_state *b);

/* The StableHLO module `text` compiled for one CPU device; it must return `outputs` tensors. `donated` are the
   inputs the text aliases to an output (`tf.aliasing_output`): a run gives up those buffers, and keeps the others.
   A program of the same text compiled lately comes from the cache of the process (32 programs), compiled once. */
an_term xla_compile(an_env *env, ng_str text, int64_t outputs, xla_list_Int donated);

/* An empty call of `p`. */
an_term xla_call(an_env *env, struct xla_Program_state *p);

/* Appends `b` as the next argument of `c`. */
an_term xla_arg(an_env *env, struct xla_Call_state *c, struct xla_Buffer_state *b);

/* Runs the program of `c` over its arguments, which must be all of them, in order: a new buffer per output. The
   call runs once: after the run it holds no buffer and no program, whatever the outputs keep it for. */
an_term xla_execute(an_env *env, struct xla_Call_state *c);

/* How many buffers are alive in the process, for a test. */
an_term xla_buffers(an_env *env);

/* How many programs the process has compiled (the misses of the cache), for a test. */
an_term xla_compiles(an_env *env);

#ifdef __cplusplus
}
#endif

#endif

/* Outside the include guard, so a file that included the header before it defined the switch still gets this
   part; its own guard keeps it to one copy. */
#if defined(XLA_NATIVE_IMPLEMENTATION) && !defined(XLA_NATIVE_IMPLEMENTATION_DONE)
#define XLA_NATIVE_IMPLEMENTATION_DONE
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

static bool xla_native_item_Int(an_env *env, an_term x, int64_t *out, ng_fail *fail) {
  return an_get_int(env, x, &(*out)) || ng_expected(fail, "Int");
}

static bool xla_native_open_list_Int(an_env *env, an_term t, xla_list_Int *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Int]");
}

bool xla_list_Int_open(an_env *env, an_term t, xla_list_Int *out) {
  ng_fail fail;
  return xla_native_open_list_Int(env, t, out, &fail) || ng_raise(env, "", "List[Int]", &fail);
}

int xla_list_Int_next(an_env *env, xla_list_Int *list, int64_t *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (xla_native_item_Int(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Int]", list->argument, at, NULL);
}

static bool xla_native_item_Float(an_env *env, an_term x, double *out, ng_fail *fail) {
  return an_get_float(env, x, &(*out)) || ng_expected(fail, "Float");
}

static bool xla_native_open_list_Float(an_env *env, an_term t, xla_list_Float *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Float]");
}

bool xla_list_Float_open(an_env *env, an_term t, xla_list_Float *out) {
  ng_fail fail;
  return xla_native_open_list_Float(env, t, out, &fail) || ng_raise(env, "", "List[Float]", &fail);
}

int xla_list_Float_next(an_env *env, xla_list_Float *list, double *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (xla_native_item_Float(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Float]", list->argument, at, NULL);
}

an_term xla_result_Void_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term xla_result_Buffer_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

void xla_list_Float_begin(an_env *env, xla_list_Float_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void xla_list_Float_push(an_env *env, xla_list_Float_builder *b, double value) {
  an_list_push(env, &b->builder, an_float(env, value));
}

an_term xla_list_Float_end(an_env *env, xla_list_Float_builder *b) { return an_list_end(env, &b->builder); }

an_term xla_result_list_Float_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term xla_result_Program_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

void xla_list_Buffer_begin(an_env *env, xla_list_Buffer_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void xla_list_Buffer_push(an_env *env, xla_list_Buffer_builder *b, an_term value) {
  an_list_push(env, &b->builder, value);
}

an_term xla_list_Buffer_end(an_env *env, xla_list_Buffer_builder *b) { return an_list_end(env, &b->builder); }

an_term xla_result_list_Buffer_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

static an_term xla_native_arch(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  (void)argv;
  return xla_arch(env);
}

static an_term xla_native_open(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_str a0;
  if (!an_get_string(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  return xla_open(env, a0);
}

static an_term xla_native_upload(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  xla_list_Int a0;
  if (!xla_native_open_list_Int(env, argv[0], &a0, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 0);
  a0.argument = 1;
  xla_list_Float a1;
  if (!xla_native_open_list_Float(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Float]", 1);
  a1.argument = 2;
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  return xla_upload(env, a0, a1, a2);
}

static an_term xla_native_load(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  ng_str a0;
  if (!an_get_string(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  xla_list_Int a2;
  if (!xla_native_open_list_Int(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 2);
  a2.argument = 3;
  int64_t a3;
  if (!an_get_int(env, argv[3], &a3)) return an_badarg(env, 3);
  return xla_load(env, a0, a1, a2, a3);
}

static an_term xla_native_from_bytes(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  ng_buf a0;
  if (!an_get_bytes(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  xla_list_Int a1;
  if (!xla_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  return xla_from_bytes(env, a0, a1, a2);
}

static an_term xla_native_filled(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  xla_list_Int a0;
  if (!xla_native_open_list_Int(env, argv[0], &a0, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 0);
  a0.argument = 1;
  double a1;
  if (!an_get_float(env, argv[1], &a1)) return an_badarg(env, 1);
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  return xla_filled(env, a0, a1, a2);
}

static an_term xla_native_readback(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct xla_Buffer_state *a0 = xla_Buffer_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return xla_readback(env, a0);
}

static an_term xla_native_compile(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  ng_str a0;
  if (!an_get_string(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  xla_list_Int a2;
  if (!xla_native_open_list_Int(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 2);
  a2.argument = 3;
  return xla_compile(env, a0, a1, a2);
}

static an_term xla_native_call(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct xla_Program_state *a0 = xla_Program_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return xla_call(env, a0);
}

static an_term xla_native_arg(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct xla_Call_state *a0 = xla_Call_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct xla_Buffer_state *a1 = xla_Buffer_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return xla_arg(env, a0, a1);
}

static an_term xla_native_execute(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct xla_Call_state *a0 = xla_Call_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return xla_execute(env, a0);
}

static an_term xla_native_buffers(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  (void)argv;
  return xla_buffers(env);
}

static an_term xla_native_compiles(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  (void)argv;
  return xla_compiles(env);
}

static const an_function xla_native_functions[] = {
  {"arch", "() -> String", xla_native_arch, AN_SHORT},
  {"open", "(String) -> Result[Void]", xla_native_open, AN_DIRTY},
  {"upload", "(List[Int], List[Float], Int) -> Result[Buffer]", xla_native_upload, AN_DIRTY},
  {"load", "(String, Int, List[Int], Int) -> Result[Buffer]", xla_native_load, AN_DIRTY},
  {"from_bytes", "(Bytes, List[Int], Int) -> Result[Buffer]", xla_native_from_bytes, AN_DIRTY},
  {"filled", "(List[Int], Float, Int) -> Result[Buffer]", xla_native_filled, AN_DIRTY},
  {"readback", "(Buffer) -> Result[List[Float]]", xla_native_readback, AN_DIRTY},
  {"compile", "(String, Int, List[Int]) -> Result[Program]", xla_native_compile, AN_DIRTY},
  {"call", "(Program) -> Call", xla_native_call, AN_SHORT},
  {"arg", "(Call, Buffer) -> Result[Void]", xla_native_arg, AN_SHORT},
  {"execute", "(Call) -> Result[List[Buffer]]", xla_native_execute, AN_DIRTY},
  {"buffers", "() -> Int", xla_native_buffers, AN_SHORT},
  {"compiles", "() -> Int", xla_native_compiles, AN_SHORT},
};

/* Ends the module's one .c file: the tables and agentica_native_init. LOAD is NULL or a const char *(*)(void)
   that runs once at the first call, and refuses the module with the message it returns. */
#define XLA_NATIVE_MODULE(LOAD) \
  static const an_resource_type xla_native_resources[] = { \
    {"Buffer", sizeof(struct xla_Buffer_state), XLA_Buffer_FREE}, \
    {"Program", sizeof(struct xla_Program_state), XLA_Program_FREE}, \
    {"Call", sizeof(struct xla_Call_state), XLA_Call_FREE}, \
  }; \
  AN_MODULE_TABLES("xla", xla_native_functions, 13, NULL, 0, xla_native_resources, 3, LOAD)
#endif
