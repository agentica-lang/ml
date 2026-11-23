/* The thin C++ layer of `src/native.agc`, on the header that native-gen makes
   from its `extern module` block. libtorch has no C API, so this file is C++
   that lives in a `.c` file: the build table of `agentica.toml` takes `.c`
   sources and names no compiler, so `CC` selects the C++ driver (see
   `tools/build-torch.sh` and `README.md`, "Local setup").

   libtorch is a system dependency, never a copy: this file touches it and
   nothing else does. Every function here is one libtorch call, the checks
   libtorch would otherwise do silently wrong (it broadcasts; the reference
   does not), and the `error` text of §9. Shapes, dtypes and devices are
   already validated in `../../torch.agc`; the checks below are the backstop
   `tensor/DESIGN.md` §9 asks a native backend for, never the only check.

   Every entry point ends in `catch (... ) { return failed(env, "op"); }`: an
   FFI call never unwinds across the C boundary, whatever the C++ side throws
   (`failed` classifies `c10::Error`, every other `std::exception`, and
   anything else) — agentica-lang/ml#12.

   Do not edit the generated header: change `src/native.agc` and make the
   header again. */

/* Every `compute def` row of the table is AN_FRESH too (API 1.5): a tensor handle owns a `torch::Tensor` of its own,
   made with `new`; a view holds its base by libtorch's refcount, never by pointing into another handle's state. So
   an arena that keeps a result keeps it alone, not the tensors it came from. The generated header says AN_DIRTY on
   every row, and native-gen has no per-function marker, so the mode is widened here for all of them: a new row must
   make a handle that owns all its memory (no `from_blob` over borrowed data, no view of memory that is not
   refcounted), or the arguments may be freed under it. */
#include "agentica_native.h"
#define AN_DIRTY (AN_DIRTY | AN_FRESH)
#define TORCH_NATIVE_IMPLEMENTATION
#include "torch_native.h"
#undef AN_DIRTY
#include "float_file.h"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

/* The state of one native tensor. The VM hands a zeroed block, and a
   `torch::Tensor` is not a C object, so the state holds a pointer to one made
   with `new` and TORCH_TensorHandle_FREE deletes it. */
struct torch_TensorHandle_state {
  torch::Tensor *value;
};

static void torch_TensorHandle_free(void *state) {
  struct torch_TensorHandle_state *self = (struct torch_TensorHandle_state *)state;

  if (self->value != NULL) {
    delete self->value;
    self->value = NULL;
  }
}
#define TORCH_TensorHandle_FREE torch_TensorHandle_free

/* The dtype codes of `src/dtypes.agc`, in the order `Dtypes.code` numbers
   them. 0..8 is F32, F16, F64, I8, I16, I32, I64, U8, Bool. */
static bool scalar_type_of(int64_t code, torch::ScalarType *out) {
  switch (code) {
    case 0: *out = torch::kFloat32; return true;
    case 1: *out = torch::kFloat16; return true;
    case 2: *out = torch::kFloat64; return true;
    case 3: *out = torch::kChar; return true;
    case 4: *out = torch::kShort; return true;
    case 5: *out = torch::kInt; return true;
    case 6: *out = torch::kLong; return true;
    case 7: *out = torch::kByte; return true;
    case 8: *out = torch::kBool; return true;
    default: return false;
  }
}

/* The name of a dtype code, for the errors of this file (the Agentica side
   names the dtypes it validates). */
static const char *dtype_name_of(int64_t code) {
  switch (code) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "F64";
    case 3: return "I8";
    case 4: return "I16";
    case 5: return "I32";
    case 6: return "I64";
    case 7: return "U8";
    case 8: return "Bool";
    default: return "unknown";
  }
}

/* The name of a libtorch dtype: `scalar_type_of` read backwards, so the two
   tables cannot drift apart. */
static std::string scalar_type_name(torch::ScalarType type) {
  torch::ScalarType other;

  for (int64_t code = 0; code < 9; code++) {
    if (scalar_type_of(code, &other) && other == type) return dtype_name_of(code);
  }
  return "unknown";
}

/* Whether a libtorch dtype holds the integers `gather` takes as positions:
   every int width, and not `Bool`. */
static bool is_index_type(torch::ScalarType type) {
  switch (type) {
    case torch::kChar:
    case torch::kShort:
    case torch::kInt:
    case torch::kLong:
    case torch::kByte:
      return true;
    default:
      return false;
  }
}

/* The integer range of a dtype code, as double: `lo` is the narrowest value it
   holds and `hi` the first one it does not, so a truncated value fits when
   `lo <= x < hi`. `to_dtype` refuses anything else, where libtorch would wrap
   it. Every bound is a double an integer of that width holds exactly. */
static bool int_range_of(int64_t code, double *lo, double *hi) {
  switch (code) {
    case 3: *lo = -128.0; *hi = 128.0; return true;
    case 4: *lo = -32768.0; *hi = 32768.0; return true;
    case 5: *lo = -2147483648.0; *hi = 2147483648.0; return true;
    case 6: *lo = -9223372036854775808.0; *hi = 9223372036854775808.0; return true;
    case 7: *lo = 0.0; *hi = 256.0; return true;
    default: return false;
  }
}

/* The range of a dtype code as the error message of `src/dtypes.agc` spells
   it, so a caller sees one text whichever side refused the value. */
static const char *int_range_text(int64_t code) {
  switch (code) {
    case 3: return "I8 holds -128..127";
    case 4: return "I16 holds -32768..32767";
    case 5: return "I32 holds -2147483648..2147483647";
    case 6: return "I64 holds -9223372036854775808..9223372036854775807";
    case 7: return "U8 holds 0..255";
    default: return "an integer dtype";
  }
}

static an_term error_text(an_env *env, const std::string &message) {
  return an_error(env, message.c_str(), message.size());
}

/* The answer of a call libtorch threw out of: `failed` is called from the
   `catch (...)` of every entry point, so one text covers a `c10::Error`, any
   other `std::exception` (a `std::bad_alloc`, say), and a throw that is neither
   (which would otherwise unwind across the C boundary). Every branch names the
   op, so a message that came from libtorch still says where it came from. */
static an_term failed(an_env *env, const char *op) {
  try {
    throw;
  } catch (const std::exception &e) {
    return error_text(env, std::string(op) + ": " + e.what());
  } catch (...) {
    return error_text(env, std::string(op) + " failed inside libtorch");
  }
}

/* A shape as the reference prints one: "[]", "[4, 3]". */
static std::string shape_text(const torch::Tensor &t) {
  std::string text = "[";
  for (int64_t i = 0; i < t.dim(); i++) {
    if (i > 0) text += ", ";
    text += std::to_string(t.size(i));
  }
  return text + "]";
}

/* A shape as a std::vector prints as its IntArrayRef does: "[4, 3]". */
static std::string dims_text(const std::vector<int64_t> &dims) {
  std::string text = "[";
  for (size_t i = 0; i < dims.size(); i++) {
    if (i > 0) text += ", ";
    text += std::to_string(dims[i]);
  }
  return text;
}

/* A binary op's shape check: libtorch broadcasts, the reference does not, so
   every pairing op refuses two shapes that are not equal before it computes. */
static bool check_same_shape(an_env *env, const torch::Tensor &left, const torch::Tensor &right, const char *op,
                             an_term *failure) {
  if (left.sizes() == right.sizes()) return true;
  *failure = error_text(env, std::string(op) + " takes two tensors of one shape, got " + shape_text(left) + " and " +
                                 shape_text(right));
  return false;
}

/* A native tensor as the resource the signature returns. `new` is the one
   allocation here that libtorch does not wrap, so a failed allocation is caught
   and reported as an error instead of unwinding across the C boundary; the
   resource made just above is then freed by the caller's arena with a NULL
   value, which the free hook handles. */
static an_term ok_tensor(an_env *env, const torch::Tensor &value) {
  an_term term;
  struct torch_TensorHandle_state *state = torch_TensorHandle_new(env, &term);

  try {
    state->value = new torch::Tensor(value);
  } catch (const std::exception &e) {
    return error_text(env, std::string("could not hold a native tensor: ") + e.what());
  } catch (...) {
    return error_text(env, "could not hold a native tensor");
  }

  return torch_result_TensorHandle_ok(env, term);
}

/* A shape argument as libtorch dimensions, refusing a negative one. */
static bool read_shape(an_env *env, torch_list_Int &shape, std::vector<int64_t> *out, an_term *failure) {
  int64_t dim;
  int code;

  while ((code = torch_list_Int_next(env, &shape, &dim)) == 1) {
    if (dim < 0) {
      *failure = error_text(env, "a shape holds no negative dimension");
      return false;
    }
    out->push_back(dim);
  }

  if (code < 0) {
    *failure = an_void(env); /* the cursor failed the call with the item's path */
    return false;
  }
  return true;
}

/* A List[Int] argument of non-negative bounds: `what` names it in the error
   text, and each caller has its own range rule on top of this one. */
static bool read_bounds(an_env *env, torch_list_Int &list, std::vector<int64_t> *out, const char *what,
                        an_term *failure) {
  int64_t bound;
  int code;

  while ((code = torch_list_Int_next(env, &list, &bound)) == 1) {
    if (bound < 0) {
      *failure = error_text(env, std::string(what) + " holds no negative value, got " + std::to_string(bound));
      return false;
    }
    out->push_back(bound);
  }

  if (code < 0) {
    *failure = an_void(env);
    return false;
  }
  return true;
}

static int64_t volume_of(const std::vector<int64_t> &dims) {
  int64_t total = 1;
  for (size_t i = 0; i < dims.size(); i++) total *= dims[i];
  return total;
}

static bool holds(const std::vector<int64_t> &axes, int64_t axis) {
  return std::find(axes.begin(), axes.end(), axis) != axes.end();
}

/* The axes of a reduction: `[]` means no reduction at all, so the result is a
   copy. libtorch reads an empty dimension list as "every dimension", which is
   why the empty case never reaches `sum`/`mean`/`max`. */
static bool read_axes(an_env *env, torch_list_Int &axes, std::vector<int64_t> *out, an_term *failure) {
  int64_t axis;
  int code;

  while ((code = torch_list_Int_next(env, &axes, &axis)) == 1) {
    if (axis < 0) {
      *failure = error_text(env, "a reduction takes no negative axis, got " + std::to_string(axis));
      return false;
    }
    out->push_back(axis);
  }

  if (code < 0) {
    *failure = an_void(env);
    return false;
  }
  return true;
}

/* The axis backstop the reductions and `squeeze` share: libtorch would answer
   a bad axis with a `c10::Error` of its own, and the count `mean` divides by
   needs it anyway, so `sum` checks it too. The range is written half-open:
   `[0, dim)`. A negative axis is refused while the list is read
   (`read_axes`). */
static bool check_axes(an_env *env, const torch::Tensor &t, const std::vector<int64_t> &axs, const char *op,
                       an_term *failure) {
  for (size_t i = 0; i < axs.size(); i++) {
    if (axs[i] >= t.dim()) {
      *failure = error_text(env, std::string(op) + " needs axes in [0, " + std::to_string(t.dim()) + "), got " +
                               std::to_string(axs[i]));
      return false;
    }
  }
  return true;
}

/* A non-finite input never reaches libtorch's transcendental core: a `NaN`
   would travel through every later op as a `NaN` instead of an `error`, and
   the reference refuses it at the same place. */
static bool all_finite(const torch::Tensor &t, an_env *env, const char *op, an_term *failure) {
  if (t.numel() == 0 || t.isfinite().all().item<bool>()) return true;
  *failure = error_text(env, std::string(op) + " needs finite values, got a non-finite one");
  return false;
}

an_term torch_create(an_env *env, torch_list_Int shape, torch_list_Float data, int64_t dtype) {
  std::vector<int64_t> dims;
  std::vector<double> values;
  torch::ScalarType scalar;
  an_term failure;
  double x;
  int code;

  if (!read_shape(env, shape, &dims, &failure)) return failure;
  if (!scalar_type_of(dtype, &scalar)) return error_text(env, "unknown dtype code " + std::to_string(dtype));

  while ((code = torch_list_Float_next(env, &data, &x)) == 1) values.push_back(x);
  if (code < 0) return an_void(env);

  int64_t wanted = volume_of(dims);
  if ((int64_t)values.size() != wanted) {
    return error_text(env, "a shape of " + std::to_string(wanted) + " values takes " +
                             std::to_string(values.size()));
  }

  try {
    /* The values arrive as doubles; an integral dtype converts exactly. */
    torch::Tensor built = torch::from_blob(values.data(), dims, torch::TensorOptions().dtype(torch::kFloat64)).clone();
    if (scalar != torch::kFloat64) built = built.to(scalar);
    return ok_tensor(env, built);
  } catch (...) {
    return failed(env, "create");
  }
}

an_term torch_load(an_env *env, ng_str path, int64_t offset, torch_list_Int shape, int64_t format) {
  std::vector<int64_t> dims;
  an_term failure;
  char why[FF_PATH_MAX + 160];

  if (!read_shape(env, shape, &dims, &failure)) return failure;

  try {
    /* The file is read into the tensor's own memory. */
    torch::Tensor built = torch::empty(dims, torch::TensorOptions().dtype(torch::kFloat32));
    if (!ff_read(path.data, path.len, offset, built.numel(), format, built.data_ptr<float>(), why, sizeof why)) {
      return error_text(env, why);
    }
    return ok_tensor(env, built);
  } catch (...) {
    return failed(env, "load");
  }
}

an_term torch_from_bytes(an_env *env, ng_buf data, torch_list_Int shape, int64_t format) {
  std::vector<int64_t> dims;
  an_term failure;
  char why[160];

  if (!read_shape(env, shape, &dims, &failure)) return failure;

  try {
    torch::Tensor built = torch::empty(dims, torch::TensorOptions().dtype(torch::kFloat32));
    if (!ff_from_bytes(data.data, data.len, built.numel(), format, built.data_ptr<float>(), why, sizeof why)) {
      return error_text(env, why);
    }
    return ok_tensor(env, built);
  } catch (...) {
    return failed(env, "from_bytes");
  }
}

an_term torch_add(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;
  an_term failure;

  if (!check_same_shape(env, left, right, "add", &failure)) return failure;

  try {
    return ok_tensor(env, left + right);
  } catch (...) {
    return failed(env, "add");
  }
}

an_term torch_sub(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;
  an_term failure;

  if (!check_same_shape(env, left, right, "sub", &failure)) return failure;

  try {
    return ok_tensor(env, left - right);
  } catch (...) {
    return failed(env, "sub");
  }
}

an_term torch_mul(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;
  an_term failure;

  if (!check_same_shape(env, left, right, "mul", &failure)) return failure;

  try {
    return ok_tensor(env, left * right);
  } catch (...) {
    return failed(env, "mul");
  }
}

/* Division, with the reference's rule: a zero divisor is an error naming the
   position, never an infinity that the next op computes on. */
an_term torch_div(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;
  an_term failure;

  if (!check_same_shape(env, left, right, "div", &failure)) return failure;

  try {
    if (right.numel() > 0) {
      torch::Tensor zeros = right.eq(0);
      if (zeros.any().item<bool>()) {
        int64_t at = zeros.reshape({-1}).nonzero().flatten()[0].item<int64_t>();
        return error_text(env, "div by zero at index " + std::to_string(at));
      }
    }
    return ok_tensor(env, left / right);
  } catch (...) {
    return failed(env, "div");
  }
}

an_term torch_neg(an_env *env, struct torch_TensorHandle_state *a) {
  try {
    return ok_tensor(env, a->value->neg());
  } catch (...) {
    return failed(env, "neg");
  }
}

an_term torch_matmul(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;

  if (left.dim() != 2 || right.dim() != 2) {
    return error_text(env, "matmul takes rank-2 tensors, got " + shape_text(left) + " and " + shape_text(right));
  }
  if (left.size(1) != right.size(0)) {
    return error_text(env, "matmul needs " + std::to_string(left.size(1)) + " in the inner dimension, got " +
                             std::to_string(right.size(0)));
  }

  try {
    return ok_tensor(env, torch::matmul(left, right));
  } catch (...) {
    return failed(env, "matmul");
  }
}

an_term torch_relu(an_env *env, struct torch_TensorHandle_state *a) {
  try {
    return ok_tensor(env, a->value->relu());
  } catch (...) {
    return failed(env, "relu");
  }
}

an_term torch_exp(an_env *env, struct torch_TensorHandle_state *a) {
  an_term failure;

  try {
    if (!all_finite(*a->value, env, "exp", &failure)) return failure;
    return ok_tensor(env, a->value->exp());
  } catch (...) {
    return failed(env, "exp");
  }
}

an_term torch_log(an_env *env, struct torch_TensorHandle_state *a) {
  const torch::Tensor &t = *a->value;
  an_term failure;

  try {
    if (!all_finite(t, env, "log", &failure)) return failure;
    if (t.numel() > 0 && !t.gt(0).all().item<bool>()) {
      return error_text(env, "log needs x > 0, got a value at or below zero");
    }
    return ok_tensor(env, t.log());
  } catch (...) {
    return failed(env, "log");
  }
}

an_term torch_sqrt(an_env *env, struct torch_TensorHandle_state *a) {
  const torch::Tensor &t = *a->value;
  an_term failure;

  try {
    if (!all_finite(t, env, "sqrt", &failure)) return failure;
    if (t.numel() > 0 && !t.ge(0).all().item<bool>()) {
      return error_text(env, "sqrt needs x >= 0, got a value below zero");
    }
    return ok_tensor(env, t.sqrt());
  } catch (...) {
    return failed(env, "sqrt");
  }
}

/* `base ** exponent`, with the reference's domain: both sides finite and the
   base positive, since the reference computes it through `exp`/`log`. */
an_term torch_pow(an_env *env, struct torch_TensorHandle_state *base, struct torch_TensorHandle_state *exponent) {
  const torch::Tensor &bs = *base->value;
  const torch::Tensor &es = *exponent->value;
  an_term failure;

  if (!check_same_shape(env, bs, es, "pow", &failure)) return failure;

  try {
    if (!all_finite(bs, env, "pow", &failure)) return failure;
    if (!all_finite(es, env, "pow", &failure)) return failure;
    if (bs.numel() > 0 && !bs.gt(0).all().item<bool>()) {
      return error_text(env, "pow needs base > 0, got a value at or below zero");
    }
    return ok_tensor(env, torch::pow(bs, es));
  } catch (...) {
    return failed(env, "pow");
  }
}

an_term torch_sigmoid(an_env *env, struct torch_TensorHandle_state *a) {
  an_term failure;

  try {
    if (!all_finite(*a->value, env, "sigmoid", &failure)) return failure;
    return ok_tensor(env, torch::sigmoid(*a->value));
  } catch (...) {
    return failed(env, "sigmoid");
  }
}

an_term torch_tanh(an_env *env, struct torch_TensorHandle_state *a) {
  an_term failure;

  try {
    if (!all_finite(*a->value, env, "tanh", &failure)) return failure;
    return ok_tensor(env, torch::tanh(*a->value));
  } catch (...) {
    return failed(env, "tanh");
  }
}

an_term torch_gelu_tanh_approx(an_env *env, struct torch_TensorHandle_state *a) {
  an_term failure;

  try {
    if (!all_finite(*a->value, env, "gelu_tanh_approx", &failure)) return failure;
    return ok_tensor(env, torch::gelu(*a->value, "tanh"));
  } catch (...) {
    return failed(env, "gelu_tanh_approx");
  }
}

an_term torch_gelu_erf(an_env *env, struct torch_TensorHandle_state *a) {
  an_term failure;

  try {
    if (!all_finite(*a->value, env, "gelu_erf", &failure)) return failure;
    return ok_tensor(env, torch::gelu(*a->value));
  } catch (...) {
    return failed(env, "gelu_erf");
  }
}

an_term torch_eq(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;
  an_term failure;

  if (!check_same_shape(env, left, right, "eq", &failure)) return failure;

  try {
    return ok_tensor(env, left.eq(right));
  } catch (...) {
    return failed(env, "eq");
  }
}

an_term torch_lt(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;
  an_term failure;

  if (!check_same_shape(env, left, right, "lt", &failure)) return failure;

  try {
    return ok_tensor(env, left.lt(right));
  } catch (...) {
    return failed(env, "lt");
  }
}

an_term torch_where(an_env *env, struct torch_TensorHandle_state *cond, struct torch_TensorHandle_state *a,
                    struct torch_TensorHandle_state *b) {
  const torch::Tensor &condition = *cond->value;
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;
  an_term failure;

  if (condition.scalar_type() != torch::kBool) {
    return error_text(env, "where needs a Bool condition, got " + scalar_type_name(condition.scalar_type()));
  }
  if (!check_same_shape(env, left, right, "where", &failure)) return failure;
  if (condition.sizes() != left.sizes()) {
    return error_text(env, "where needs a condition of shape " + shape_text(left) + ", got " + shape_text(condition));
  }

  try {
    return ok_tensor(env, torch::where(condition, left, right));
  } catch (...) {
    return failed(env, "where");
  }
}

an_term torch_clamp(an_env *env, struct torch_TensorHandle_state *a, double lo, double hi) {
  if (lo > hi) return error_text(env, "clamp needs lo <= hi, got two bounds out of order");

  try {
    return ok_tensor(env, a->value->clamp(lo, hi));
  } catch (...) {
    return failed(env, "clamp");
  }
}

an_term torch_sum(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes, bool keep_dims) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> axs;
  an_term failure;

  if (!read_axes(env, axes, &axs, &failure)) return failure;
  if (!check_axes(env, t, axs, "sum", &failure)) return failure;

  try {
    if (axs.empty()) return ok_tensor(env, t.clone());
    return ok_tensor(env, t.sum(axs, keep_dims));
  } catch (...) {
    return failed(env, "sum");
  }
}

an_term torch_mean(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes, bool keep_dims) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> axs;
  an_term failure;

  if (!read_axes(env, axes, &axs, &failure)) return failure;

  if (!check_axes(env, t, axs, "mean", &failure)) return failure;

  int64_t count = 1;
  for (size_t i = 0; i < axs.size(); i++) count *= t.size(axs[i]);
  if (count == 0) return error_text(env, "mean of zero elements is an error");

  try {
    if (axs.empty()) return ok_tensor(env, t.clone());
    return ok_tensor(env, t.mean(axs, keep_dims));
  } catch (...) {
    return failed(env, "mean");
  }
}

/* The largest value over `axes`, through `amax`: an empty tensor is an error
   here as it is in the reference, where libtorch would answer `-inf`. */
an_term torch_max(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes, bool keep_dims) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> axs;
  an_term failure;

  if (!read_axes(env, axes, &axs, &failure)) return failure;
  if (!check_axes(env, t, axs, "max", &failure)) return failure;
  if (t.numel() == 0) return error_text(env, "max of zero elements is an error");

  try {
    if (axs.empty()) return ok_tensor(env, t.clone());
    return ok_tensor(env, t.amax(axs, keep_dims));
  } catch (...) {
    return failed(env, "max");
  }
}

/* The first position of the largest value over `axes`, as `I64`.
   libtorch reduces one axis at a time, so the reduced axes are permuted to the
   end (in the order given, which is the row-major order the reference numbers
   their flat offset by), flattened into one axis, and `argmax` runs there once;
   CPU `argmax` keeps the lowest position on a tie, as the reference's `>` does.
   `[]` reduces nothing: the reference answers zeros of the input's shape. */
an_term torch_argmax(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes, bool keep_dims) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> axs;
  an_term failure;

  if (!read_axes(env, axes, &axs, &failure)) return failure;
  if (!check_axes(env, t, axs, "argmax", &failure)) return failure;
  if (t.numel() == 0) return error_text(env, "argmax of zero elements is an error");

  try {
    if (axs.empty()) return ok_tensor(env, torch::zeros(t.sizes(), t.options().dtype(torch::kLong)));

    std::vector<int64_t> order;
    std::vector<int64_t> out;
    int64_t inner = 1;
    for (int64_t i = 0; i < t.dim(); i++) {
      if (!holds(axs, i)) order.push_back(i);
    }
    for (size_t i = 0; i < axs.size(); i++) order.push_back(axs[i]);
    for (int64_t i = 0; i < t.dim(); i++) {
      if (holds(axs, i)) {
        inner *= t.size(i);
        if (keep_dims) out.push_back(1);
      } else {
        out.push_back(t.size(i));
      }
    }

    torch::Tensor moved = t.permute(order);
    torch::Tensor flat = moved.reshape({t.numel() / inner, inner});
    return ok_tensor(env, flat.argmax(1).reshape(out));
  } catch (...) {
    return failed(env, "argmax");
  }
}

an_term torch_reshape(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int shape) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> dims;
  an_term failure;

  if (!read_shape(env, shape, &dims, &failure)) return failure;

  int64_t wanted = volume_of(dims);
  if (wanted != t.numel()) {
    return error_text(env, "reshape needs " + std::to_string(wanted) + " values for " + shape_text(*a->value) +
                             ", got " + std::to_string(t.numel()));
  }

  try {
    return ok_tensor(env, t.reshape(dims));
  } catch (...) {
    return failed(env, "reshape");
  }
}

/* Drops the size-1 axes `axes` names, or every size-1 axis for `[]`. The size
   backstop comes first, so a refused axis leaves no half-squeezed tensor. */
an_term torch_squeeze(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> axs;
  an_term failure;

  if (!read_axes(env, axes, &axs, &failure)) return failure;
  if (!check_axes(env, t, axs, "squeeze", &failure)) return failure;

  for (size_t i = 0; i < axs.size(); i++) {
    if (t.size(axs[i]) != 1) {
      return error_text(env, "squeeze needs a dimension of size 1 at " + std::to_string(axs[i]) + ", got " +
                               std::to_string(t.size(axs[i])));
    }
  }

  try {
    if (axs.empty()) return ok_tensor(env, t.squeeze());
    torch::Tensor out = t;
    for (size_t i = 0; i < axs.size(); i++) out = out.squeeze(axs[i]);
    return ok_tensor(env, out);
  } catch (...) {
    return failed(env, "squeeze");
  }
}

an_term torch_unsqueeze(an_env *env, struct torch_TensorHandle_state *a, int64_t axis) {
  const torch::Tensor &t = *a->value;

  if (axis < 0 || axis > t.dim()) {
    return error_text(env, "unsqueeze needs an axis in [0, " + std::to_string(t.dim()) + "], got " +
                             std::to_string(axis));
  }

  try {
    return ok_tensor(env, t.unsqueeze(axis));
  } catch (...) {
    return failed(env, "unsqueeze");
  }
}

/* The `[starts[i], ends[i])` window per axis, contiguous so the readback is the
   row-major order of the window and not of the buffer it views. */
an_term torch_slice(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int starts, torch_list_Int ends) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> from;
  std::vector<int64_t> to;
  an_term failure;

  if (!read_bounds(env, starts, &from, "slice", &failure)) return failure;
  if (!read_bounds(env, ends, &to, "slice", &failure)) return failure;

  if ((int64_t)from.size() != t.dim() || (int64_t)to.size() != t.dim()) {
    return error_text(env, "slice needs one start and one end per axis of " + shape_text(t) + ", got " + dims_text(from) +
                             " and " + dims_text(to));
  }
  for (int64_t i = 0; i < t.dim(); i++) {
    if (to[i] < from[i] || to[i] > t.size(i)) {
      return error_text(env, "slice needs 0 <= start <= end <= " + std::to_string(t.size(i)) + ", got " +
                               std::to_string(from[i]) + ".." + std::to_string(to[i]));
    }
  }

  try {
    torch::Tensor out = t;
    for (int64_t i = 0; i < t.dim(); i++) out = out.narrow(i, from[i], to[i] - from[i]);
    return ok_tensor(env, out.contiguous());
  } catch (...) {
    return failed(env, "slice");
  }
}

/* Two tensors joined along `axis`. The signature is binary because a
   `List[TensorHandle]` cannot cross this boundary; `src/torch.agc` folds. */
an_term torch_concat(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b,
                     int64_t axis) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;

  if (left.dim() != right.dim()) {
    return error_text(env, "concat needs one rank, got " + shape_text(left) + " and " + shape_text(right));
  }
  if (left.scalar_type() != right.scalar_type()) {
    return error_text(env, "concat needs one dtype, got " + scalar_type_name(left.scalar_type()) + " and " +
                             scalar_type_name(right.scalar_type()));
  }
  if (axis < 0 || axis >= left.dim()) {
    return error_text(env, "concat needs an axis in [0, " + std::to_string(left.dim()) + "), got " +
                             std::to_string(axis));
  }
  for (int64_t i = 0; i < left.dim(); i++) {
    if (i != axis && left.size(i) != right.size(i)) {
      return error_text(env, "concat needs dimension " + std::to_string(i) + " of size " +
                               std::to_string(left.size(i)) + ", got " + std::to_string(right.size(i)));
    }
  }

  try {
    return ok_tensor(env, torch::cat({left, right}, axis));
  } catch (...) {
    return failed(env, "concat");
  }
}

/* `t` widened to `shape`: the trailing-1 rule the wrapper checks, re-checked
   here because `expand` would answer a bad one with its own error text. */
an_term torch_broadcast_to(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int shape) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> dims;
  an_term failure;

  if (!read_shape(env, shape, &dims, &failure)) return failure;
  if ((int64_t)dims.size() < t.dim()) {
    return error_text(env, "broadcast_to cannot drop axes, got " + shape_text(t) + " into " + dims_text(dims));
  }

  int64_t lead = (int64_t)dims.size() - t.dim();
  for (int64_t i = 0; i < t.dim(); i++) {
    int64_t want = dims[lead + i];
    if (t.size(i) != want && t.size(i) != 1) {
      return error_text(env, "broadcast_to needs " + std::to_string(t.size(i)) + " to stay " +
                               std::to_string(t.size(i)) + " or widen from 1, got " + std::to_string(want));
    }
  }

  try {
    return ok_tensor(env, t.expand(dims).contiguous());
  } catch (...) {
    return failed(env, "broadcast_to");
  }
}

/* The rows of `t` at the integer tensor `indices`: `indices.dims ++
   t.dims[1:]`. Every index is range-checked first, so an out-of-range one is
   this file's error and not libtorch's. */
an_term torch_gather(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *indices) {
  const torch::Tensor &source = *a->value;
  const torch::Tensor &where = *indices->value;

  if (source.dim() < 1) {
    return error_text(env, "gather needs a rank >= 1 source, got " + shape_text(source));
  }
  if (!is_index_type(where.scalar_type())) {
    return error_text(env, "gather needs integer indices, got " + scalar_type_name(where.scalar_type()));
  }

  try {
    torch::Tensor flat = where.to(torch::kLong).reshape({-1});
    int64_t rows = source.size(0);
    if (flat.numel() > 0) {
      int64_t lowest = flat.min().item<int64_t>();
      int64_t highest = flat.max().item<int64_t>();
      if (lowest < 0 || highest >= rows) {
        int64_t worst = lowest < 0 ? lowest : highest;
        return error_text(env, "gather needs indices in [0, " + std::to_string(rows) + "), got " + std::to_string(worst));
      }
    }

    std::vector<int64_t> out;
    for (int64_t i = 0; i < where.dim(); i++) out.push_back(where.size(i));
    for (int64_t i = 1; i < source.dim(); i++) out.push_back(source.size(i));
    return ok_tensor(env, source.index_select(0, flat).reshape(out));
  } catch (...) {
    return failed(env, "gather");
  }
}

an_term torch_transpose(an_env *env, struct torch_TensorHandle_state *a) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> dims;

  try {
    if (t.dim() < 2) return ok_tensor(env, t.clone());
    for (int64_t i = t.dim() - 1; i >= 0; i--) dims.push_back(i);
    /* `.contiguous()` so the readback is the row-major order of the result,
       which is what the reference produces. */
    return ok_tensor(env, t.permute(dims).contiguous());
  } catch (...) {
    return failed(env, "transpose");
  }
}

an_term torch_to_dtype(an_env *env, struct torch_TensorHandle_state *a, int64_t dtype) {
  const torch::Tensor &t = *a->value;
  torch::ScalarType target;

  if (!scalar_type_of(dtype, &target)) return error_text(env, "unknown dtype code " + std::to_string(dtype));

  try {
    if (t.numel() == 0) return ok_tensor(env, t.to(target));
    if (target == torch::kBool) return ok_tensor(env, t.to(torch::kBool));

    double lo, hi;
    if (int_range_of(dtype, &lo, &hi)) {
      /* Truncation towards zero is what `to_dtype` promises, so the check is
         on the truncated value: 127.9 fits I8 as 127, as in the reference.
         The extremes are read in double, which is what every dtype of this
         backend converts through. */
      torch::Tensor probe = t.scalar_type() == torch::kFloat64 ? t : t.to(torch::kFloat64);
      double lowest = std::trunc(probe.min().item<double>());
      double highest = std::trunc(probe.max().item<double>());
      if (!std::isfinite(lowest) || !std::isfinite(highest)) {
        return error_text(env, std::string("to_dtype needs finite values for ") + dtype_name_of(dtype));
      }
      if (lowest < lo || highest >= hi) {
        char number[32];
        snprintf(number, sizeof number, "%g", lowest < lo ? lowest : highest);
        return error_text(env, std::string(int_range_text(dtype)) + ", got " + number);
      }
    }
    return ok_tensor(env, t.to(target));
  } catch (...) {
    return failed(env, "to_dtype");
  }
}

an_term torch_copy(an_env *env, struct torch_TensorHandle_state *a) {
  try {
    return ok_tensor(env, a->value->clone());
  } catch (...) {
    return failed(env, "copy");
  }
}

an_term torch_contiguous(an_env *env, struct torch_TensorHandle_state *a) {
  try {
    return ok_tensor(env, a->value->contiguous());
  } catch (...) {
    return failed(env, "contiguous");
  }
}

/* The window a convolution or a pooling moves: `stride` positions per step,
   `padding` zeros on each side, no dilation. All of it is re-checked here
   because libtorch's `conv1d`/`avg_pool1d` would otherwise reinterpret a
   `[C, L]` input as a batch of channels, or answer a zero-length window. */
static bool check_window(an_env *env, const torch::Tensor &t, int64_t kernel, int64_t stride, int64_t padding,
                         const char *op, int64_t length, an_term *failure) {
  if (stride < 1) {
    *failure = error_text(env, std::string(op) + " needs stride >= 1, got " + std::to_string(stride));
    return false;
  }
  if (padding < 0) {
    *failure = error_text(env, std::string(op) + " needs padding >= 0, got " + std::to_string(padding));
    return false;
  }
  if (length < 1) {
    *failure = error_text(env, std::string(op) + " needs a nonempty input, got " + shape_text(t));
    return false;
  }
  if (length + 2 * padding < kernel) {
    *failure = error_text(env, std::string(op) + " needs a window of " + std::to_string(kernel) + " positions in " +
                                     std::to_string(length) + " padded to " + std::to_string(length + 2 * padding));
    return false;
  }
  return true;
}

an_term torch_conv1d(an_env *env, struct torch_TensorHandle_state *input, struct torch_TensorHandle_state *weight,
                     int64_t stride, int64_t padding) {
  const torch::Tensor &x = *input->value;
  const torch::Tensor &w = *weight->value;
  an_term failure;

  if (x.dim() != 2 && x.dim() != 3) {
    return error_text(env, "conv1d needs a rank-2 or rank-3 input, got " + shape_text(x));
  }
  if (w.dim() != 3) return error_text(env, "conv1d needs a rank-3 weight, got " + shape_text(w));
  if (!check_window(env, x, w.size(2), stride, padding, "conv1d", x.size(x.dim() - 1), &failure)) return failure;
  if (w.size(1) != x.size(x.dim() - 2)) {
    return error_text(env, "conv1d needs " + std::to_string(x.size(x.dim() - 2)) + " input channels, got " +
                             std::to_string(w.size(1)));
  }
  if (x.scalar_type() != w.scalar_type()) {
    return error_text(env, "conv1d needs one dtype, got " + scalar_type_name(x.scalar_type()) + " and " +
                             scalar_type_name(w.scalar_type()));
  }

  try {
    /* libtorch's `conv1d` takes a batch axis; a rank-2 input gains one and
       loses it again in the answer, which is the shape the reference builds. */
    torch::Tensor in = x.dim() == 3 ? x : x.unsqueeze(0);
    torch::Tensor out = torch::conv1d(in, w, torch::Tensor(), {stride}, {padding});
    return ok_tensor(env, x.dim() == 3 ? out : out.squeeze(0));
  } catch (...) {
    return failed(env, "conv1d");
  }
}

/* The mean over a window of `kernel` positions. Padding stops at half the
   kernel, which is libtorch's own rule: refusing it here keeps the message the
   same width as every other one in this file, and `src/tensor.agc` refuses the
   same windows before dispatch, so the reference never accepts what this
   backend cannot run. */
an_term torch_avg_pool1d(an_env *env, struct torch_TensorHandle_state *input, int64_t kernel, int64_t stride,
                         int64_t padding) {
  const torch::Tensor &x = *input->value;
  an_term failure;

  if (x.dim() != 2 && x.dim() != 3) {
    return error_text(env, "avg_pool1d needs a rank-2 or rank-3 input, got " + shape_text(x));
  }
  if (kernel < 1) return error_text(env, "avg_pool1d needs kernel >= 1, got " + std::to_string(kernel));
  if (!check_window(env, x, kernel, stride, padding, "avg_pool1d", x.size(x.dim() - 1), &failure)) return failure;
  if (padding > kernel / 2) {
    return error_text(env, "avg_pool1d needs padding <= " + std::to_string(kernel / 2) + " for a kernel of " +
                             std::to_string(kernel) + ", got " + std::to_string(padding));
  }

  try {
    torch::Tensor in = x.dim() == 3 ? x : x.unsqueeze(0);
    torch::Tensor out = torch::avg_pool1d(in, {kernel}, {stride}, {padding});
    return ok_tensor(env, x.dim() == 3 ? out : out.squeeze(0));
  } catch (...) {
    return failed(env, "avg_pool1d");
  }
}

an_term torch_readback(an_env *env, struct torch_TensorHandle_state *a) {
  try {
    torch::Tensor flat = a->value->contiguous();
    if (flat.scalar_type() != torch::kDouble) flat = flat.to(torch::kDouble);
    int64_t n = flat.numel();
    const double *values = n > 0 ? flat.data_ptr<double>() : NULL;

    torch_list_Float_builder out;
    torch_list_Float_begin(env, &out, (size_t)n);
    for (int64_t i = 0; i < n; i++) torch_list_Float_push(env, &out, values[i]);
    return torch_result_list_Float_ok(env, torch_list_Float_end(env, &out));
  } catch (...) {
    return failed(env, "readback");
  }
}

TORCH_NATIVE_MODULE(NULL)
