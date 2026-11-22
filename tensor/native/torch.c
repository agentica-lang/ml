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

   Do not edit the generated header: change `src/native.agc` and make the
   header again. */
#define TORCH_NATIVE_IMPLEMENTATION
#include "torch_native.h"

#include <torch/torch.h>

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

/* A shape as the reference prints one: "[]", "[4, 3]". */
static std::string shape_text(const torch::Tensor &t) {
  std::string text = "[";
  for (int64_t i = 0; i < t.dim(); i++) {
    if (i > 0) text += ", ";
    text += std::to_string(t.size(i));
  }
  return text + "]";
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

static int64_t volume_of(const std::vector<int64_t> &dims) {
  int64_t total = 1;
  for (size_t i = 0; i < dims.size(); i++) total *= dims[i];
  return total;
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
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

an_term torch_add(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;

  if (left.sizes() != right.sizes()) {
    return error_text(env, "add takes two tensors of one shape, got " + shape_text(left) + " and " +
                             shape_text(right));
  }

  try {
    return ok_tensor(env, left + right);
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

an_term torch_mul(an_env *env, struct torch_TensorHandle_state *a, struct torch_TensorHandle_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;

  if (left.sizes() != right.sizes()) {
    return error_text(env, "mul takes two tensors of one shape, got " + shape_text(left) + " and " +
                             shape_text(right));
  }

  try {
    return ok_tensor(env, left * right);
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
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
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

an_term torch_relu(an_env *env, struct torch_TensorHandle_state *a) {
  try {
    return ok_tensor(env, a->value->relu());
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

/* A non-finite input never reaches libtorch's transcendental core: a `NaN`
   would travel through every later op as a `NaN` instead of an `error`, and
   the reference refuses it at the same place. */
static bool all_finite(const torch::Tensor &t, an_env *env, const char *op, an_term *failure) {
  if (t.numel() == 0 || t.isfinite().all().item<bool>()) return true;
  *failure = error_text(env, std::string(op) + " needs finite values, got a non-finite one");
  return false;
}

an_term torch_exp(an_env *env, struct torch_TensorHandle_state *a) {
  an_term failure;

  try {
    if (!all_finite(*a->value, env, "exp", &failure)) return failure;
    return ok_tensor(env, a->value->exp());
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
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
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

/* The axes of a reduction: `[]` means no reduction at all, so the result is a
   copy. libtorch reads an empty dimension list as "every dimension", which is
   why the empty case never reaches `sum`/`mean`. */
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

/* The axis backstop both reductions share: libtorch would answer a bad axis
   with a `c10::Error` of its own, and the count `mean` divides by needs it
   anyway, so `sum` checks it too. The range is written half-open: `[0, dim)`.
   A negative axis is refused while the list is read (`read_axes`). */
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

an_term torch_sum(an_env *env, struct torch_TensorHandle_state *a, torch_list_Int axes, bool keep_dims) {
  const torch::Tensor &t = *a->value;
  std::vector<int64_t> axs;
  an_term failure;

  if (!read_axes(env, axes, &axs, &failure)) return failure;
  if (!check_axes(env, t, axs, "sum", &failure)) return failure;

  try {
    if (axs.empty()) return ok_tensor(env, t.clone());
    return ok_tensor(env, t.sum(axs, keep_dims));
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
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
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
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
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
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
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
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
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
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
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

TORCH_NATIVE_MODULE(NULL)
