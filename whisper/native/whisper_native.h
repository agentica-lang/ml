/* The C side of `extern module "whisper"` in whisper.agc, made by native-gen.
   Do not edit it: change the .agc and make it again. */
#ifndef WHISPER_NATIVE_H
#define WHISPER_NATIVE_H

#include "agentica_native.h"

#if AN_API_MAJOR != 1
#error "whisper_native.h needs version 1 of the native API (agentica_native.h)"
#endif
#if AN_API_MINOR < 2
#error "whisper_native.h needs version 1.2 or later of the native API (agentica_native.h)"
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
enum { WHISPER_RESOURCE_Model = 0 };

/* A loaded model: a resource whose state the native side owns.

   Define it, and WHISPER_Model_FREE (a void (*)(void *state), or NULL), before WHISPER_NATIVE_MODULE.
   A parameter of this type is its state. _new makes one, with zeroed state. _get gives the state of a term
   that is an argument of this call, and NULL for any other, one made by _new in the same call included. The
   state is valid until the call returns: do not keep the pointer. */
struct whisper_Model_state;
static inline struct whisper_Model_state *whisper_Model_new(an_env *env, an_term *out) {
  return (struct whisper_Model_state *)an_resource_new(env, WHISPER_RESOURCE_Model, out);
}
static inline struct whisper_Model_state *whisper_Model_get(an_env *env, an_term t) {
  void *state;
  return an_get_resource(env, t, WHISPER_RESOURCE_Model, &state) ? (struct whisper_Model_state *)state : NULL;
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
} whisper_list_Float;
bool whisper_list_Float_open(an_env *env, an_term t, whisper_list_Float *out);
int whisper_list_Float_next(an_env *env, whisper_list_Float *list, double *out);

an_term whisper_result_Model_ok(an_env *env, an_term value);

an_term whisper_result_String_ok(an_env *env, ng_str value);

/* Loads the model at `path` (a ggml/gguf file); error when it cannot be read. */
an_term whisper_load_model(an_env *env, ng_str path);

/* Transcribes 16 kHz mono PCM audio; error when the model cannot run it. */
an_term whisper_transcribe(an_env *env, struct whisper_Model_state *model, whisper_list_Float pcm);

#ifdef __cplusplus
}
#endif

#endif

/* Outside the include guard, so a file that included the header before it defined the switch still gets this
   part; its own guard keeps it to one copy. */
#if defined(WHISPER_NATIVE_IMPLEMENTATION) && !defined(WHISPER_NATIVE_IMPLEMENTATION_DONE)
#define WHISPER_NATIVE_IMPLEMENTATION_DONE
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

static bool whisper_native_item_Float(an_env *env, an_term x, double *out, ng_fail *fail) {
  return an_get_float(env, x, &(*out)) || ng_expected(fail, "Float");
}

static bool whisper_native_open_list_Float(an_env *env, an_term t, whisper_list_Float *out, ng_fail *fail) {
  memset(out, 0, sizeof *out);
  out->term = t;
  if (an_list_cursor_begin(env, t, &out->cursor, &out->len)) return true;
  out->failed = true;
  return ng_expected(fail, "List[Float]");
}

bool whisper_list_Float_open(an_env *env, an_term t, whisper_list_Float *out) {
  ng_fail fail;
  return whisper_native_open_list_Float(env, t, out, &fail) || ng_raise(env, "", "List[Float]", &fail);
}

int whisper_list_Float_next(an_env *env, whisper_list_Float *list, double *out) {
  an_term x;
  ng_fail fail;
  size_t at = list->at;
  if (list->failed) return -1;
  if (!an_list_next(env, &list->cursor, &x)) return 0;
  list->at++;
  if (whisper_native_item_Float(env, x, out, &fail)) return 1;
  list->failed = true;
  return ng_fail_cursor_item(env, &fail, "List[Float]", list->argument, at, NULL);
}

an_term whisper_result_Model_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term whisper_result_String_ok(an_env *env, ng_str value) {
  return an_ok(env, an_string(env, value.data, value.len));
}

static an_term whisper_native_load_model(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_str a0;
  if (!an_get_string(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  return whisper_load_model(env, a0);
}

static an_term whisper_native_transcribe(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct whisper_Model_state *a0 = whisper_Model_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  whisper_list_Float a1;
  if (!whisper_native_open_list_Float(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Float]", 1);
  a1.argument = 2;
  return whisper_transcribe(env, a0, a1);
}

static const an_function whisper_native_functions[] = {
  {"load_model", "(String) -> Result[Model]", whisper_native_load_model, AN_SHORT},
  {"transcribe", "(Model, List[Float]) -> Result[String]", whisper_native_transcribe, AN_DIRTY},
};

/* Ends the module's one .c file: the tables and agentica_native_init. LOAD is NULL or a const char *(*)(void)
   that runs once at the first call, and refuses the module with the message it returns. */
#define WHISPER_NATIVE_MODULE(LOAD) \
  static const an_resource_type whisper_native_resources[] = { \
    {"Model", sizeof(struct whisper_Model_state), WHISPER_Model_FREE}, \
  }; \
  AN_MODULE_TABLES("whisper", whisper_native_functions, 2, NULL, 0, whisper_native_resources, 1, LOAD)
#endif
