/* The C side of `extern module "mathx"` in mathx.agc, made by native-gen.
   Do not edit it: change the .agc and make it again. */
#ifndef MATHX_NATIVE_H
#define MATHX_NATIVE_H

#include "agentica_native.h"

#if AN_API_MAJOR != 1
#error "mathx_native.h needs version 1 of the native API (agentica_native.h)"
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

an_term mathx_exp(an_env *env, double x);

an_term mathx_log(an_env *env, double x);

an_term mathx_log10(an_env *env, double x);

an_term mathx_sin(an_env *env, double x);

an_term mathx_cos(an_env *env, double x);

#ifdef __cplusplus
}
#endif

#endif

/* Outside the include guard, so a file that included the header before it defined the switch still gets this
   part; its own guard keeps it to one copy. */
#if defined(MATHX_NATIVE_IMPLEMENTATION) && !defined(MATHX_NATIVE_IMPLEMENTATION_DONE)
#define MATHX_NATIVE_IMPLEMENTATION_DONE
/* Declared with C linkage, so a module built as C++ exports it under its C name. */
#ifdef __cplusplus
extern "C" {
#endif
AN_EXPORT const an_module *agentica_native_init(const an_api *api);
#ifdef __cplusplus
}
#endif

static an_term mathx_native_exp(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  double a0;
  if (!an_get_float(env, argv[0], &a0)) return an_badarg(env, 0);
  return mathx_exp(env, a0);
}

static an_term mathx_native_log(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  double a0;
  if (!an_get_float(env, argv[0], &a0)) return an_badarg(env, 0);
  return mathx_log(env, a0);
}

static an_term mathx_native_log10(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  double a0;
  if (!an_get_float(env, argv[0], &a0)) return an_badarg(env, 0);
  return mathx_log10(env, a0);
}

static an_term mathx_native_sin(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  double a0;
  if (!an_get_float(env, argv[0], &a0)) return an_badarg(env, 0);
  return mathx_sin(env, a0);
}

static an_term mathx_native_cos(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  double a0;
  if (!an_get_float(env, argv[0], &a0)) return an_badarg(env, 0);
  return mathx_cos(env, a0);
}

static const an_function mathx_native_functions[] = {
  {"exp", "(Float) -> Float", mathx_native_exp, AN_SHORT},
  {"log", "(Float) -> Float", mathx_native_log, AN_SHORT},
  {"log10", "(Float) -> Float", mathx_native_log10, AN_SHORT},
  {"sin", "(Float) -> Float", mathx_native_sin, AN_SHORT},
  {"cos", "(Float) -> Float", mathx_native_cos, AN_SHORT},
};

/* Ends the module's one .c file: the tables and agentica_native_init. LOAD is NULL or a const char *(*)(void)
   that runs once at the first call, and refuses the module with the message it returns. */
#define MATHX_NATIVE_MODULE(LOAD) AN_MODULE_TABLES("mathx", mathx_native_functions, 5, NULL, 0, NULL, 0, LOAD)
#endif
