/* The C side of `extern module "torchspike"` in torch_spike.agc, made by native-gen.
   Do not edit it: change the .agc and make it again. */
#ifndef TORCHSPIKE_NATIVE_H
#define TORCHSPIKE_NATIVE_H

#include "agentica_native.h"

#if AN_API_MAJOR != 1
#error "torchspike_native.h needs version 1 of the native API (agentica_native.h)"
#endif
#if AN_API_MINOR < 2
#error "torchspike_native.h needs version 1.2 or later of the native API (agentica_native.h)"
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
enum { TORCHSPIKE_RESOURCE_TorchTensor = 0 };

/* A tensor on the libtorch CPU backend. The native state owns the
   `torch::Tensor`; nothing here can name a shape, a dtype or a device.

   Define it, and TORCHSPIKE_TorchTensor_FREE (a void (*)(void *state), or NULL), before TORCHSPIKE_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct torchspike_TorchTensor_state;
static inline struct torchspike_TorchTensor_state *torchspike_TorchTensor_new(an_env *env, an_term *out) {
  return (struct torchspike_TorchTensor_state *)an_resource_new(env, TORCHSPIKE_RESOURCE_TorchTensor, out);
}
static inline struct torchspike_TorchTensor_state *torchspike_TorchTensor_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, TORCHSPIKE_RESOURCE_TorchTensor, &state) ? (struct torchspike_TorchTensor_state *)state : NULL;
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
} torchspike_list_Int;
bool torchspike_list_Int_open(an_env *env, an_term t, torchspike_list_Int *out);
int torchspike_list_Int_next(an_env *env, torchspike_list_Int *list, int64_t *out);

typedef struct {
  an_term term;
  an_list_cursor cursor;
  size_t len;
  size_t at;
  bool failed;
  int argument; /* 1 + the index of the argument of the call it was opened for, or 0 (_open) */
} torchspike_list_Float;
bool torchspike_list_Float_open(an_env *env, an_term t, torchspike_list_Float *out);
int torchspike_list_Float_next(an_env *env, torchspike_list_Float *list, double *out);

an_term torchspike_result_TorchTensor_ok(an_env *env, an_term value);

an_term torchspike_result_Float_ok(an_env *env, double value);

/* A tensor of `shape` from row-major `data`. The values cross as doubles
   and are stored as `float32`, the dtype libtorch computes in. */
an_term torchspike_create(an_env *env, torchspike_list_Int shape, torchspike_list_Float data);

/* Elementwise `a + b`, both of one shape. */
an_term torchspike_add(an_env *env, struct torchspike_TorchTensor_state *a, struct torchspike_TorchTensor_state *b);

/* The rank-2 product of `[rows, inner]` and `[inner, cols]`. */
an_term torchspike_matmul(an_env *env, struct torchspike_TorchTensor_state *a, struct torchspike_TorchTensor_state *b);

/* `max(0, x)`, elementwise. */
an_term torchspike_relu(an_env *env, struct torchspike_TorchTensor_state *a);

/* The sum of every element, read back across the boundary as one `Float`. */
an_term torchspike_sum(an_env *env, struct torchspike_TorchTensor_state *a);

#ifdef __cplusplus
}
#endif

#endif

/* Outside the include guard, so a file that included the header before it defined the switch still gets this
   part; its own guard keeps it to one copy. */
#if defined(TORCHSPIKE_NATIVE_IMPLEMENTATION) && !defined(TORCHSPIKE_NATIVE_IMPLEMENTATION_DONE)
#define TORCHSPIKE_NATIVE_IMPLEMENTATION_DONE
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

static bool torchspike_native_item_Int(an_env *env, an_term x, int64_t *out, ng_fail *fail) {
  return an_get_int(env, x, &(*out)) || ng_expected(fail, "Int");
}

static bool torchspike_native_open_list_Int(an_env *env, an_term t, torchspike_list_Int *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Int]");
}

bool torchspike_list_Int_open(an_env *env, an_term t, torchspike_list_Int *out) {
  ng_fail fail;
  return torchspike_native_open_list_Int(env, t, out, &fail) || ng_raise(env, "", "List[Int]", &fail);
}

int torchspike_list_Int_next(an_env *env, torchspike_list_Int *list, int64_t *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (torchspike_native_item_Int(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Int]", list->argument, at, NULL);
}

static bool torchspike_native_item_Float(an_env *env, an_term x, double *out, ng_fail *fail) {
  return an_get_float(env, x, &(*out)) || ng_expected(fail, "Float");
}

static bool torchspike_native_open_list_Float(an_env *env, an_term t, torchspike_list_Float *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Float]");
}

bool torchspike_list_Float_open(an_env *env, an_term t, torchspike_list_Float *out) {
  ng_fail fail;
  return torchspike_native_open_list_Float(env, t, out, &fail) || ng_raise(env, "", "List[Float]", &fail);
}

int torchspike_list_Float_next(an_env *env, torchspike_list_Float *list, double *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (torchspike_native_item_Float(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Float]", list->argument, at, NULL);
}

an_term torchspike_result_TorchTensor_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term torchspike_result_Float_ok(an_env *env, double value) {
  return an_ok(env, an_float(env, value));
}

static an_term torchspike_native_create(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  torchspike_list_Int a0;
  if (!torchspike_native_open_list_Int(env, argv[0], &a0, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 0);
  a0.argument = 1;
  torchspike_list_Float a1;
  if (!torchspike_native_open_list_Float(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Float]", 1);
  a1.argument = 2;
  return torchspike_create(env, a0, a1);
}

static an_term torchspike_native_add(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torchspike_TorchTensor_state *a0 = torchspike_TorchTensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torchspike_TorchTensor_state *a1 = torchspike_TorchTensor_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torchspike_add(env, a0, a1);
}

static an_term torchspike_native_matmul(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torchspike_TorchTensor_state *a0 = torchspike_TorchTensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torchspike_TorchTensor_state *a1 = torchspike_TorchTensor_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torchspike_matmul(env, a0, a1);
}

static an_term torchspike_native_relu(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torchspike_TorchTensor_state *a0 = torchspike_TorchTensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torchspike_relu(env, a0);
}

static an_term torchspike_native_sum(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torchspike_TorchTensor_state *a0 = torchspike_TorchTensor_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torchspike_sum(env, a0);
}

static const an_function torchspike_native_functions[] = {
  {"create", "(List[Int], List[Float]) -> Result[TorchTensor]", torchspike_native_create, AN_DIRTY},
  {"add", "(TorchTensor, TorchTensor) -> Result[TorchTensor]", torchspike_native_add, AN_DIRTY},
  {"matmul", "(TorchTensor, TorchTensor) -> Result[TorchTensor]", torchspike_native_matmul, AN_DIRTY},
  {"relu", "(TorchTensor) -> Result[TorchTensor]", torchspike_native_relu, AN_DIRTY},
  {"sum", "(TorchTensor) -> Result[Float]", torchspike_native_sum, AN_DIRTY},
};

/* Ends the module's one .c file: the tables and agentica_native_init. LOAD is NULL or a const char *(*)(void)
   that runs once at the first call, and refuses the module with the message it returns. */
#define TORCHSPIKE_NATIVE_MODULE(LOAD) \
  static const an_resource_type torchspike_native_resources[] = { \
    {"TorchTensor", sizeof(struct torchspike_TorchTensor_state), TORCHSPIKE_TorchTensor_FREE}, \
  }; \
  AN_MODULE_TABLES("torchspike", torchspike_native_functions, 5, NULL, 0, torchspike_native_resources, 1, LOAD)
#endif
