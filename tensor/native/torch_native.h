/* The C side of `extern module "torch"` in native.agc, made by native-gen.
   Do not edit it: change the .agc and make it again. */
#ifndef TORCH_NATIVE_H
#define TORCH_NATIVE_H

#include "agentica_native.h"

#if AN_API_MAJOR != 1
#error "torch_native.h needs version 1 of the native API (agentica_native.h)"
#endif
#if AN_API_MINOR < 2
#error "torch_native.h needs version 1.2 or later of the native API (agentica_native.h)"
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
enum { TORCH_RESOURCE_TensorHandle = 0 };

/* A tensor on the libtorch CPU backend. The state owns a `torch::Tensor`
   made with `new`; `native/torch.c` deletes it in the free hook.

   Define it, and TORCH_TensorHandle_FREE (a void (*)(void *state), or NULL), before TORCH_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct torch_TensorHandle_state;
static inline struct torch_TensorHandle_state *torch_TensorHandle_new(an_env *env, an_term *out) {
  return (struct torch_TensorHandle_state *)an_resource_new(env, TORCH_RESOURCE_TensorHandle, out);
}
static inline struct torch_TensorHandle_state *torch_TensorHandle_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, TORCH_RESOURCE_TensorHandle, &state) ? (struct torch_TensorHandle_state *)state : NULL;
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
} torch_list_Int;
bool torch_list_Int_open(an_env *env, an_term t, torch_list_Int *out);
int torch_list_Int_next(an_env *env, torch_list_Int *list, int64_t *out);

typedef struct {
  an_term term;
  an_list_cursor cursor;
  size_t len;
  size_t at;
  bool failed;
  int argument; /* 1 + the index of the argument of the call it was opened for, or 0 (_open) */
} torch_list_Float;
bool torch_list_Float_open(an_env *env, an_term t, torch_list_Float *out);
int torch_list_Float_next(an_env *env, torch_list_Float *list, double *out);

an_term torch_result_TensorHandle_ok(an_env *env, an_term value);

typedef struct {
  an_list_builder builder;
} torch_list_Float_builder;
void torch_list_Float_begin(an_env *env, torch_list_Float_builder *b, size_t hint);
void torch_list_Float_push(an_env *env, torch_list_Float_builder *b, double value);
an_term torch_list_Float_end(an_env *env, torch_list_Float_builder *b);

an_term torch_result_list_Float_ok(an_env *env, an_term value);

/* A tensor of `shape` from row-major `data` in the dtype `dtype` names
   (`TorchDtypes.code`). The values cross as doubles; an integer dtype must hold
   integral values in range, which the callers of `create` check first. */
an_term torch_create(an_env *env, torch_list_Int shape, torch_list_Float data, int64_t dtype);

/* An `F32` tensor of `shape` from the little-endian floats of `format` (a
   code of `Types.file_code`) that start at byte `offset` of the file `path`.
   C reads the file into the tensor's own memory, so no value crosses as a
   list; a range past the end of the file is an error. */
an_term torch_load(an_env *env, ng_str path, int64_t offset, torch_list_Int shape, int64_t format);

/* The same from `data`, the bytes of the range that Agentica read
   (`TENSOR_LOAD=agentica`): the call reads them in place and widens them into
   the tensor's own memory. `data` must be exactly the range, or the call is
   an error. */
an_term torch_from_bytes(an_env *env, ng_buf data, torch_list_Int shape, int64_t format);

/* Elementwise `a + b`, two tensors of one shape. */
an_term torch_add(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b);

/* Elementwise `a * b`, two tensors of one shape. */
an_term torch_mul(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b);

/* Elementwise `a - b`, two tensors of one shape. */
an_term torch_sub(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b);

/* Elementwise `a / b`, two tensors of one shape; a zero divisor is an
   error, never an infinity that travels on as a value. */
an_term torch_div(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b);

/* Elementwise `-x`. */
an_term torch_neg(an_env *env, struct torch_TensorHandle_state *a);

/* The rank-2 product `[rows, inner] . [inner, cols]`. */
an_term torch_matmul(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b);

/* `max(0, x)`, elementwise. */
an_term torch_relu(an_env *env, struct torch_TensorHandle_state *a);

/* `exp(x)`, elementwise, finite inputs only: a non-finite value is an
   error here rather than a `NaN` in the result. */
an_term torch_exp(an_env *env, struct torch_TensorHandle_state *a);

/* `log(x)`, elementwise, finite positive inputs only. */
an_term torch_log(an_env *env, struct torch_TensorHandle_state *a);

/* `sqrt(x)`, elementwise, finite inputs with `x >= 0` only. */
an_term torch_sqrt(an_env *env, struct torch_TensorHandle_state *a);

/* `base ** exponent`, elementwise over two tensors of one shape, finite
   inputs with `base > 0` only. */
an_term torch_pow(an_env *env, struct torch_TensorHandle_state *base, struct torch_TensorHandle_state *exponent);

/* Sigmoid, elementwise, finite inputs only. */
an_term torch_sigmoid(an_env *env, struct torch_TensorHandle_state *a);

/* Tanh, elementwise, finite inputs only. */
an_term torch_tanh(an_env *env, struct torch_TensorHandle_state *a);

/* The tanh approximation of GELU, elementwise, finite inputs only. */
an_term torch_gelu_tanh_approx(an_env *env, struct torch_TensorHandle_state *a);

/* The exact (erf) GELU, elementwise, finite inputs only. */
an_term torch_gelu_erf(an_env *env, struct torch_TensorHandle_state *a);

/* `a == b` elementwise as a `Bool` tensor, two tensors of one shape. */
an_term torch_eq(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b);

/* `a < b` elementwise as a `Bool` tensor, two tensors of one shape. */
an_term torch_lt(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b);

/* `cond ? a : b`: a `Bool` condition and the two same-shape, same-dtype
   tensors it picks between. */
an_term torch_where(an_env *env, struct torch_TensorHandle_state *cond, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b);

/* Elementwise clamp to `[lo, hi]`, `lo <= hi`. */
an_term torch_clamp(an_env *env, struct torch_TensorHandle_state *a, double lo, double hi);

/* The sum over `axes`, dropping them or keeping them as `1` as `keep_dims`
   asks. `[]` is no reduction at all, never "every axis". */
an_term torch_sum(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes, bool keep_dims);

/* The mean over `axes`, same conventions as `sum`; zero reduced elements
   is an error, as in the reference, not a `NaN`. */
an_term torch_mean(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes, bool keep_dims);

/* The largest value over `axes`, same conventions as `sum`; an empty tensor
   is an error, as the reference says, never `-inf`. */
an_term torch_max(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes, bool keep_dims);

/* The first position of the largest value over `axes`, as `I64`; `[]`
   reduces nothing and answers zeros, as the reference does. */
an_term torch_argmax(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes, bool keep_dims);

/* The same values with `shape`; the element counts must agree. */
an_term torch_reshape(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int shape);

/* Drops the size-1 axes `axes` names, or every size-1 axis for `[]`. */
an_term torch_squeeze(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes);

/* Inserts a size-1 axis at `axis`. */
an_term torch_unsqueeze(an_env *env, struct torch_TensorHandle_state *a, int64_t axis);

/* The `[starts[i], ends[i])` window per axis, row-major in the result. */
an_term torch_slice(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int starts, torch_list_Int ends);

/* `a` and `b` joined along `axis`: one rank, one dtype, every other
   dimension equal. */
an_term torch_concat(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b, int64_t axis);

/* `a` widened to `shape`: trailing axes align, size-1 axes widen. */
an_term torch_broadcast_to(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int shape);

/* The rows of `a` at the integer tensor `indices`:
   `indices.dims ++ a.dims[1:]`. */
an_term torch_gather(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *indices);

/* Reverses every axis, as the reference's `transpose` does (rank 2 swaps
   rows and columns; below rank 2 nothing changes). */
an_term torch_transpose(an_env *env, struct torch_TensorHandle_state *a);

/* `a` in the dtype `dtype` names: float to int truncates towards zero and
   range-checks, anything nonzero becomes `true`, an int width that cannot
   hold a value is an error. */
an_term torch_to_dtype(an_env *env, struct torch_TensorHandle_state *a, int64_t dtype);

/* A copy that shares no buffer with `a`. */
an_term torch_copy(an_env *env, struct torch_TensorHandle_state *a);

/* `a` in row-major order, copied when it is not already. */
an_term torch_contiguous(an_env *env, struct torch_TensorHandle_state *a);

/* The cross-correlation of `input` with `weight` over the last axis, moved
   by `stride`, with `padding` zeros on each side. */
an_term torch_conv1d(an_env *env, struct torch_TensorHandle_state *input, struct torch_TensorHandle_state *weight, int64_t stride, int64_t padding);

/* The mean over a window of `kernel` positions, moved by `stride`, with
   `padding` zeros on each side and a divisor of `kernel`. */
an_term torch_avg_pool1d(an_env *env, struct torch_TensorHandle_state *input, int64_t kernel, int64_t stride, int64_t padding);

/* Host readback: the row-major values as storage doubles, which the caller
   tags with the tensor's dtype. `I64` values past 2^53 lose precision. */
an_term torch_readback(an_env *env, struct torch_TensorHandle_state *a);

#ifdef __cplusplus
}
#endif

#endif

/* Outside the include guard, so a file that included the header before it defined the switch still gets this
   part; its own guard keeps it to one copy. */
#if defined(TORCH_NATIVE_IMPLEMENTATION) && !defined(TORCH_NATIVE_IMPLEMENTATION_DONE)
#define TORCH_NATIVE_IMPLEMENTATION_DONE
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

static bool torch_native_item_Int(an_env *env, an_term x, int64_t *out, ng_fail *fail) {
  return an_get_int(env, x, &(*out)) || ng_expected(fail, "Int");
}

static bool torch_native_open_list_Int(an_env *env, an_term t, torch_list_Int *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Int]");
}

bool torch_list_Int_open(an_env *env, an_term t, torch_list_Int *out) {
  ng_fail fail;
  return torch_native_open_list_Int(env, t, out, &fail) || ng_raise(env, "", "List[Int]", &fail);
}

int torch_list_Int_next(an_env *env, torch_list_Int *list, int64_t *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (torch_native_item_Int(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Int]", list->argument, at, NULL);
}

static bool torch_native_item_Float(an_env *env, an_term x, double *out, ng_fail *fail) {
  return an_get_float(env, x, &(*out)) || ng_expected(fail, "Float");
}

static bool torch_native_open_list_Float(an_env *env, an_term t, torch_list_Float *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Float]");
}

bool torch_list_Float_open(an_env *env, an_term t, torch_list_Float *out) {
  ng_fail fail;
  return torch_native_open_list_Float(env, t, out, &fail) || ng_raise(env, "", "List[Float]", &fail);
}

int torch_list_Float_next(an_env *env, torch_list_Float *list, double *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (torch_native_item_Float(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Float]", list->argument, at, NULL);
}

an_term torch_result_TensorHandle_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

void torch_list_Float_begin(an_env *env, torch_list_Float_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void torch_list_Float_push(an_env *env, torch_list_Float_builder *b, double value) {
  an_list_push(env, &b->builder, an_float(env, value));
}

an_term torch_list_Float_end(an_env *env, torch_list_Float_builder *b) { return an_list_end(env, &b->builder); }

an_term torch_result_list_Float_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

static an_term torch_native_create(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  torch_list_Int a0;
  if (!torch_native_open_list_Int(env, argv[0], &a0, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 0);
  a0.argument = 1;
  torch_list_Float a1;
  if (!torch_native_open_list_Float(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Float]", 1);
  a1.argument = 2;
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  return torch_create(env, a0, a1, a2);
}

static an_term torch_native_load(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  ng_str a0;
  if (!an_get_string(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  torch_list_Int a2;
  if (!torch_native_open_list_Int(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 2);
  a2.argument = 3;
  int64_t a3;
  if (!an_get_int(env, argv[3], &a3)) return an_badarg(env, 3);
  return torch_load(env, a0, a1, a2, a3);
}

static an_term torch_native_from_bytes(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  ng_buf a0;
  if (!an_get_bytes(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  torch_list_Int a1;
  if (!torch_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  return torch_from_bytes(env, a0, a1, a2);
}

static an_term torch_native_add(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torch_add(env, a0, a1);
}

static an_term torch_native_mul(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torch_mul(env, a0, a1);
}

static an_term torch_native_sub(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torch_sub(env, a0, a1);
}

static an_term torch_native_div(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torch_div(env, a0, a1);
}

static an_term torch_native_neg(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_neg(env, a0);
}

static an_term torch_native_matmul(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torch_matmul(env, a0, a1);
}

static an_term torch_native_relu(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_relu(env, a0);
}

static an_term torch_native_exp(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_exp(env, a0);
}

static an_term torch_native_log(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_log(env, a0);
}

static an_term torch_native_sqrt(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_sqrt(env, a0);
}

static an_term torch_native_pow(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torch_pow(env, a0, a1);
}

static an_term torch_native_sigmoid(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_sigmoid(env, a0);
}

static an_term torch_native_tanh(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_tanh(env, a0);
}

static an_term torch_native_gelu_tanh_approx(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_gelu_tanh_approx(env, a0);
}

static an_term torch_native_gelu_erf(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_gelu_erf(env, a0);
}

static an_term torch_native_eq(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torch_eq(env, a0, a1);
}

static an_term torch_native_lt(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torch_lt(env, a0, a1);
}

static an_term torch_native_where(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  struct torch_TensorHandle_state *a2 = torch_TensorHandle_get(env, argv[2]);
  if (!a2) return an_badarg(env, 2);
  return torch_where(env, a0, a1, a2);
}

static an_term torch_native_clamp(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  double a1;
  if (!an_get_float(env, argv[1], &a1)) return an_badarg(env, 1);
  double a2;
  if (!an_get_float(env, argv[2], &a2)) return an_badarg(env, 2);
  return torch_clamp(env, a0, a1, a2);
}

static an_term torch_native_sum(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  torch_list_Int a1;
  if (!torch_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  bool a2;
  if (!an_get_bool(env, argv[2], &a2)) return an_badarg(env, 2);
  return torch_sum(env, a0, a1, a2);
}

static an_term torch_native_mean(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  torch_list_Int a1;
  if (!torch_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  bool a2;
  if (!an_get_bool(env, argv[2], &a2)) return an_badarg(env, 2);
  return torch_mean(env, a0, a1, a2);
}

static an_term torch_native_max(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  torch_list_Int a1;
  if (!torch_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  bool a2;
  if (!an_get_bool(env, argv[2], &a2)) return an_badarg(env, 2);
  return torch_max(env, a0, a1, a2);
}

static an_term torch_native_argmax(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  torch_list_Int a1;
  if (!torch_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  bool a2;
  if (!an_get_bool(env, argv[2], &a2)) return an_badarg(env, 2);
  return torch_argmax(env, a0, a1, a2);
}

static an_term torch_native_reshape(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  torch_list_Int a1;
  if (!torch_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  return torch_reshape(env, a0, a1);
}

static an_term torch_native_squeeze(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  torch_list_Int a1;
  if (!torch_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  return torch_squeeze(env, a0, a1);
}

static an_term torch_native_unsqueeze(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  return torch_unsqueeze(env, a0, a1);
}

static an_term torch_native_slice(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  torch_list_Int a1;
  if (!torch_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  torch_list_Int a2;
  if (!torch_native_open_list_Int(env, argv[2], &a2, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 2);
  a2.argument = 3;
  return torch_slice(env, a0, a1, a2);
}

static an_term torch_native_concat(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  return torch_concat(env, a0, a1, a2);
}

static an_term torch_native_broadcast_to(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  torch_list_Int a1;
  if (!torch_native_open_list_Int(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Int]", 1);
  a1.argument = 2;
  return torch_broadcast_to(env, a0, a1);
}

static an_term torch_native_gather(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  return torch_gather(env, a0, a1);
}

static an_term torch_native_transpose(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_transpose(env, a0);
}

static an_term torch_native_to_dtype(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  return torch_to_dtype(env, a0, a1);
}

static an_term torch_native_copy(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_copy(env, a0);
}

static an_term torch_native_contiguous(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_contiguous(env, a0);
}

static an_term torch_native_conv1d(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  struct torch_TensorHandle_state *a1 = torch_TensorHandle_get(env, argv[1]);
  if (!a1) return an_badarg(env, 1);
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  int64_t a3;
  if (!an_get_int(env, argv[3], &a3)) return an_badarg(env, 3);
  return torch_conv1d(env, a0, a1, a2, a3);
}

static an_term torch_native_avg_pool1d(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  int64_t a1;
  if (!an_get_int(env, argv[1], &a1)) return an_badarg(env, 1);
  int64_t a2;
  if (!an_get_int(env, argv[2], &a2)) return an_badarg(env, 2);
  int64_t a3;
  if (!an_get_int(env, argv[3], &a3)) return an_badarg(env, 3);
  return torch_avg_pool1d(env, a0, a1, a2, a3);
}

static an_term torch_native_readback(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  struct torch_TensorHandle_state *a0 = torch_TensorHandle_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  return torch_readback(env, a0);
}

static const an_function torch_native_functions[] = {
  {"create", "(List[Int], List[Float], Int) -> Result[TensorHandle]", torch_native_create, AN_DIRTY},
  {"load", "(String, Int, List[Int], Int) -> Result[TensorHandle]", torch_native_load, AN_DIRTY},
  {"from_bytes", "(Bytes, List[Int], Int) -> Result[TensorHandle]", torch_native_from_bytes, AN_DIRTY},
  {"add", "(TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_add, AN_DIRTY},
  {"mul", "(TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_mul, AN_DIRTY},
  {"sub", "(TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_sub, AN_DIRTY},
  {"div", "(TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_div, AN_DIRTY},
  {"neg", "(TensorHandle) -> Result[TensorHandle]", torch_native_neg, AN_DIRTY},
  {"matmul", "(TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_matmul, AN_DIRTY},
  {"relu", "(TensorHandle) -> Result[TensorHandle]", torch_native_relu, AN_DIRTY},
  {"exp", "(TensorHandle) -> Result[TensorHandle]", torch_native_exp, AN_DIRTY},
  {"log", "(TensorHandle) -> Result[TensorHandle]", torch_native_log, AN_DIRTY},
  {"sqrt", "(TensorHandle) -> Result[TensorHandle]", torch_native_sqrt, AN_DIRTY},
  {"pow", "(TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_pow, AN_DIRTY},
  {"sigmoid", "(TensorHandle) -> Result[TensorHandle]", torch_native_sigmoid, AN_DIRTY},
  {"tanh", "(TensorHandle) -> Result[TensorHandle]", torch_native_tanh, AN_DIRTY},
  {"gelu_tanh_approx", "(TensorHandle) -> Result[TensorHandle]", torch_native_gelu_tanh_approx, AN_DIRTY},
  {"gelu_erf", "(TensorHandle) -> Result[TensorHandle]", torch_native_gelu_erf, AN_DIRTY},
  {"eq", "(TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_eq, AN_DIRTY},
  {"lt", "(TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_lt, AN_DIRTY},
  {"where", "(TensorHandle, TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_where, AN_DIRTY},
  {"clamp", "(TensorHandle, Float, Float) -> Result[TensorHandle]", torch_native_clamp, AN_DIRTY},
  {"sum", "(TensorHandle, List[Int], Bool) -> Result[TensorHandle]", torch_native_sum, AN_DIRTY},
  {"mean", "(TensorHandle, List[Int], Bool) -> Result[TensorHandle]", torch_native_mean, AN_DIRTY},
  {"max", "(TensorHandle, List[Int], Bool) -> Result[TensorHandle]", torch_native_max, AN_DIRTY},
  {"argmax", "(TensorHandle, List[Int], Bool) -> Result[TensorHandle]", torch_native_argmax, AN_DIRTY},
  {"reshape", "(TensorHandle, List[Int]) -> Result[TensorHandle]", torch_native_reshape, AN_DIRTY},
  {"squeeze", "(TensorHandle, List[Int]) -> Result[TensorHandle]", torch_native_squeeze, AN_DIRTY},
  {"unsqueeze", "(TensorHandle, Int) -> Result[TensorHandle]", torch_native_unsqueeze, AN_DIRTY},
  {"slice", "(TensorHandle, List[Int], List[Int]) -> Result[TensorHandle]", torch_native_slice, AN_DIRTY},
  {"concat", "(TensorHandle, TensorHandle, Int) -> Result[TensorHandle]", torch_native_concat, AN_DIRTY},
  {"broadcast_to", "(TensorHandle, List[Int]) -> Result[TensorHandle]", torch_native_broadcast_to, AN_DIRTY},
  {"gather", "(TensorHandle, TensorHandle) -> Result[TensorHandle]", torch_native_gather, AN_DIRTY},
  {"transpose", "(TensorHandle) -> Result[TensorHandle]", torch_native_transpose, AN_DIRTY},
  {"to_dtype", "(TensorHandle, Int) -> Result[TensorHandle]", torch_native_to_dtype, AN_DIRTY},
  {"copy", "(TensorHandle) -> Result[TensorHandle]", torch_native_copy, AN_DIRTY},
  {"contiguous", "(TensorHandle) -> Result[TensorHandle]", torch_native_contiguous, AN_DIRTY},
  {"conv1d", "(TensorHandle, TensorHandle, Int, Int) -> Result[TensorHandle]", torch_native_conv1d, AN_DIRTY},
  {"avg_pool1d", "(TensorHandle, Int, Int, Int) -> Result[TensorHandle]", torch_native_avg_pool1d, AN_DIRTY},
  {"readback", "(TensorHandle) -> Result[List[Float]]", torch_native_readback, AN_DIRTY},
};

/* Ends the module's one .c file: the tables and agentica_native_init. LOAD is NULL or a const char *(*)(void)
   that runs once at the first call, and refuses the module with the message it returns. */
#define TORCH_NATIVE_MODULE(LOAD) \
  static const an_resource_type torch_native_resources[] = { \
    {"TensorHandle", sizeof(struct torch_TensorHandle_state), TORCH_TensorHandle_FREE}, \
  }; \
  AN_MODULE_TABLES("torch", torch_native_functions, 40, NULL, 0, torch_native_resources, 1, LOAD)
#endif
