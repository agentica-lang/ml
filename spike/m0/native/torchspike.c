/* The thin C++ layer of torch_spike.agc, on the header that native-gen makes
   from its `extern module` block. libtorch has no C API, so this file is C++
   that lives in a `.c` file: the build table of `agentica.toml` takes `.c`
   sources and names no compiler, so `CC` selects the C++ driver (see
   `tools/build-torch.sh` and `README.md`, "The native side").

   libtorch is a system dependency, never a copy: `props()` below is the only
   place this file touches it, and it is four operations. Do not edit the
   generated header: change the .agc and make the header again. */
#define TORCHSPIKE_NATIVE_IMPLEMENTATION
#include "torchspike_native.h"

#include <torch/torch.h>

#include <string>
#include <vector>

/* The state of one native tensor. The VM hands a zeroed block, and a
   `torch::Tensor` is not a C object, so the state holds a pointer to one made
   with `new` and TORCHSPIKE_TorchTensor_FREE deletes it. */
struct torchspike_TorchTensor_state {
  torch::Tensor *value;
};

static void torchspike_TorchTensor_free(void *state) {
  struct torchspike_TorchTensor_state *self = (struct torchspike_TorchTensor_state *)state;

  if (self->value != NULL) {
    delete self->value;
    self->value = NULL;
  }
}
#define TORCHSPIKE_TorchTensor_FREE torchspike_TorchTensor_free

/* An op that fails returns `error(text)`, as the reference backend does; the
   message has to be built here, because the boundary carries no error type
   and no kind (see the README's gap list). */
static an_term error_text(an_env *env, const std::string &message) {
  return an_error(env, message.c_str(), message.size());
}

/* A shape as the reference prints one: "[]", "[4, 3]". The native side has it
   only inside the tensor: the boundary carries no shape, so every message
   that names one rebuilds it here. */
static std::string shape_text(const torch::Tensor &t) {
  std::string text = "[";
  for (size_t i = 0; i < (size_t)t.dim(); i++) {
    if (i > 0) text += ", ";
    text += std::to_string(t.size((int64_t)i));
  }
  return text + "]";
}

/* A native tensor as the resource that the signature returns. */
static an_term ok_tensor(an_env *env, const torch::Tensor &value) {
  an_term term;
  struct torchspike_TorchTensor_state *state = torchspike_TorchTensor_new(env, &term);

  state->value = new torch::Tensor(value);
  return torchspike_result_TorchTensor_ok(env, term);
}

/* The one line of libtorch that the whole spike stands on: creation from
   row-major values and a shape. The values arrive as doubles and become
   float32 (libtorch's default dtype here), so the boundary's `List[Float]`
   says nothing about what the tensor holds. */
an_term torchspike_create(an_env *env, torchspike_list_Int shape, torchspike_list_Float data) {
  std::vector<int64_t> dims;
  std::vector<float> values;
  int64_t dim;
  double x;
  int code;

  while ((code = torchspike_list_Int_next(env, &shape, &dim)) == 1) {
    if (dim < 0) return error_text(env, "a shape holds no negative dimension");
    dims.push_back(dim);
  }
  if (code < 0) return an_void(env); /* the cursor failed the call with the item's path */

  int64_t wanted = 1;
  for (size_t i = 0; i < dims.size(); i++) wanted *= dims[i];

  while ((code = torchspike_list_Float_next(env, &data, &x)) == 1) values.push_back((float)x);
  if (code < 0) return an_void(env);
  if ((int64_t)values.size() != wanted)
    return error_text(env, "a shape of " + std::to_string(wanted) + " values takes " +
                             std::to_string(values.size()));

  try {
    return ok_tensor(env, torch::from_blob(values.data(), dims, torch::kFloat32).clone());
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

/* libtorch broadcasts, the reference does not; the check is here because the
   boundary cannot carry the shapes for the Agentica wrapper to check with
   (`tensor/DESIGN.md` §9 asks for exactly that check, before dispatch). */
an_term torchspike_add(an_env *env, struct torchspike_TorchTensor_state *a, struct torchspike_TorchTensor_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;

  if (left.sizes() != right.sizes())
    return error_text(env, "add takes two tensors of one shape, got " + shape_text(left) + " and " + shape_text(right));

  try {
    return ok_tensor(env, left + right);
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

an_term torchspike_matmul(an_env *env, struct torchspike_TorchTensor_state *a,
                          struct torchspike_TorchTensor_state *b) {
  const torch::Tensor &left = *a->value;
  const torch::Tensor &right = *b->value;

  if (left.dim() != 2 || right.dim() != 2)
    return error_text(env, "matmul takes rank-2 tensors, got " + shape_text(left) + " and " + shape_text(right));
  if (left.size(1) != right.size(0))
    return error_text(env, "matmul needs " + std::to_string(left.size(1)) + " in the inner dimension, got " +
                             std::to_string(right.size(0)));

  try {
    return ok_tensor(env, torch::matmul(left, right));
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

an_term torchspike_relu(an_env *env, struct torchspike_TorchTensor_state *a) {
  try {
    return ok_tensor(env, a->value->relu());
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

/* The sum of every element, read back as one Float: the scalar result of the
   demo's loss crosses as a value, not as a rank-0 tensor (README, gaps). */
an_term torchspike_sum(an_env *env, struct torchspike_TorchTensor_state *a) {
  try {
    return torchspike_result_Float_ok(env, a->value->sum().item<double>());
  } catch (const c10::Error &e) {
    return error_text(env, e.what());
  }
}

TORCHSPIKE_NATIVE_MODULE(NULL)
