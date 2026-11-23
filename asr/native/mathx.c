/* `extern module "mathx"` of mathx.agc: libm's exp, log, log10, sin and cos.
   Temporary, until Agentica's Math has them (agentica-lang/agentica#1925). */
#define MATHX_NATIVE_IMPLEMENTATION
#include "mathx_native.h"

#include <math.h>

an_term mathx_exp(an_env *env, double x) { return an_float(env, exp(x)); }
an_term mathx_log(an_env *env, double x) { return an_float(env, log(x)); }
an_term mathx_log10(an_env *env, double x) { return an_float(env, log10(x)); }
an_term mathx_sin(an_env *env, double x) { return an_float(env, sin(x)); }
an_term mathx_cos(an_env *env, double x) { return an_float(env, cos(x)); }

MATHX_NATIVE_MODULE(NULL)
