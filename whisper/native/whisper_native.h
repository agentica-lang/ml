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

/* The rows of the type table: the TYPE of an_struct, an_enum and their getters. */
enum { WHISPER_TYPE_Options = 0, WHISPER_TYPE_Segment = 1, WHISPER_TYPE_Transcript = 2, WHISPER_TYPE_SegmentDetail = 3, WHISPER_TYPE_Detection = 4, WHISPER_TYPE_Token = 5 };

/* The fields of Options, in declared order. */
enum { WHISPER_Options_language = 0, WHISPER_Options_translate = 1, WHISPER_Options_beam_size = 2, WHISPER_Options_token_timestamps = 3, WHISPER_Options_max_len = 4, WHISPER_Options_language_probabilities = 5 };

/* The fields of Segment, in declared order. */
enum { WHISPER_Segment_start_ms = 0, WHISPER_Segment_end_ms = 1, WHISPER_Segment_text = 2 };

/* The fields of Transcript, in declared order. */
enum { WHISPER_Transcript_language = 0, WHISPER_Transcript_segments = 1, WHISPER_Transcript_detection = 2 };

/* The fields of SegmentDetail, in declared order. */
enum { WHISPER_SegmentDetail_start_ms = 0, WHISPER_SegmentDetail_end_ms = 1, WHISPER_SegmentDetail_text = 2, WHISPER_SegmentDetail_tokens = 3, WHISPER_SegmentDetail_avg_logprob = 4, WHISPER_SegmentDetail_no_speech_probability = 5 };

/* The fields of Detection, in declared order. */
enum { WHISPER_Detection_language = 0, WHISPER_Detection_probability = 1, WHISPER_Detection_probabilities = 2 };

/* The fields of Token, in declared order. */
enum { WHISPER_Token_ids = 0, WHISPER_Token_text = 1, WHISPER_Token_probability = 2, WHISPER_Token_log_probability = 3, WHISPER_Token_start_ms = 4, WHISPER_Token_end_ms = 5 };

/* The rows of the resource table: the TYPE of an_resource_new and an_get_resource. */
enum { WHISPER_RESOURCE_Model = 0 };

/* A loaded model: a resource whose state the native side owns (the
   whisper context plus the mutex that serializes inference on it).

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

/* The structs and enums as C values. A String or Bytes in them is borrowed, a term lasts for the call: keep them
   in C locals. _get reads a term into OUT; false when it is not one, and then the call has failed with the path of
   the first bad part. A function whose signature does not reach the type reads none ("Point: expected Point").
   _make and the constructors (one per variant) make a term. */

typedef struct {
  ng_str language;
  bool translate;
  int64_t beam_size;
  bool token_timestamps;
  int64_t max_len;
  bool language_probabilities;
} whisper_Options;
bool whisper_Options_get(an_env *env, an_term t, whisper_Options *out);

typedef struct {
  int64_t start_ms;
  int64_t end_ms;
  ng_str text;
} whisper_Segment;
an_term whisper_Segment_make(an_env *env, const whisper_Segment *value);

typedef struct {
  ng_str language;
  an_term segments;
  an_term detection;
} whisper_Transcript;
an_term whisper_Transcript_make(an_env *env, const whisper_Transcript *value);

typedef struct {
  int64_t start_ms;
  int64_t end_ms;
  ng_str text;
  an_term tokens;
  double avg_logprob;
  double no_speech_probability;
} whisper_SegmentDetail;
an_term whisper_SegmentDetail_make(an_env *env, const whisper_SegmentDetail *value);

typedef struct {
  ng_str language;
  double probability;
  an_term probabilities;
} whisper_Detection;
an_term whisper_Detection_make(an_env *env, const whisper_Detection *value);

typedef struct {
  an_term ids;
  ng_str text;
  double probability;
  double log_probability;
  int64_t start_ms;
  int64_t end_ms;
} whisper_Token;
an_term whisper_Token_make(an_env *env, const whisper_Token *value);

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

typedef struct {
  an_list_builder builder;
} whisper_list_Segment_builder;
void whisper_list_Segment_begin(an_env *env, whisper_list_Segment_builder *b, size_t hint);
void whisper_list_Segment_push(an_env *env, whisper_list_Segment_builder *b, whisper_Segment value);
an_term whisper_list_Segment_end(an_env *env, whisper_list_Segment_builder *b);

an_term whisper_result_list_Segment_ok(an_env *env, an_term value);

an_term whisper_result_Transcript_ok(an_env *env, whisper_Transcript value);

an_term whisper_optional_String_some(an_env *env, ng_str value);

typedef struct {
  an_list_builder builder;
} whisper_list_SegmentDetail_builder;
void whisper_list_SegmentDetail_begin(an_env *env, whisper_list_SegmentDetail_builder *b, size_t hint);
void whisper_list_SegmentDetail_push(an_env *env, whisper_list_SegmentDetail_builder *b, whisper_SegmentDetail value);
an_term whisper_list_SegmentDetail_end(an_env *env, whisper_list_SegmentDetail_builder *b);

an_term whisper_optional_Detection_some(an_env *env, whisper_Detection value);

typedef struct {
  an_list_builder builder;
} whisper_list_Token_builder;
void whisper_list_Token_begin(an_env *env, whisper_list_Token_builder *b, size_t hint);
void whisper_list_Token_push(an_env *env, whisper_list_Token_builder *b, whisper_Token value);
an_term whisper_list_Token_end(an_env *env, whisper_list_Token_builder *b);

typedef struct {
  an_term map;
} whisper_map_String_Float_builder;
void whisper_map_String_Float_begin(an_env *env, whisper_map_String_Float_builder *b);
void whisper_map_String_Float_put(an_env *env, whisper_map_String_Float_builder *b, ng_str key, double value);
an_term whisper_map_String_Float_end(an_env *env, whisper_map_String_Float_builder *b);

typedef struct {
  an_list_builder builder;
} whisper_list_Int_builder;
void whisper_list_Int_begin(an_env *env, whisper_list_Int_builder *b, size_t hint);
void whisper_list_Int_push(an_env *env, whisper_list_Int_builder *b, int64_t value);
an_term whisper_list_Int_end(an_env *env, whisper_list_Int_builder *b);

/* Loads the model at `path` (a ggml/gguf file); error when it cannot be read. */
an_term whisper_load_model(an_env *env, ng_str path);

/* Transcribes 16 kHz mono PCM audio (at most five minutes: longer input is
   an error, so one call is bounded) as `options` asks, returning the
   segments the engine emitted, in time order; error when the model cannot
   run it. */
an_term whisper_transcribe_with(an_env *env, struct whisper_Model_state *model, whisper_list_Float pcm, whisper_Options options);

/* Like `transcribe_with`, with each segment's tokens and probabilities,
   the language decoded and, when `options` asks, the language detected. */
an_term whisper_transcribe_detailed(an_env *env, struct whisper_Model_state *model, whisper_list_Float pcm, whisper_Options options);

/* The English name of a language code (`"es"` is `"spanish"`); none for a
   code the engine does not know. */
an_term whisper_language_name(an_env *env, ng_str code);

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

static bool whisper_native_get_Options(an_env *env, an_term t, whisper_Options *out, ng_fail *fail) {
  an_term x;
  if (!an_get_field(env, t, WHISPER_TYPE_Options, 0, &x)) return ng_expected(fail, "Options");
  if (!an_get_string(env, x, &out->language.data, &out->language.len)) return ng_leaf(fail, "String", "language");
  if (!an_get_field(env, t, WHISPER_TYPE_Options, 1, &x)) return ng_expected(fail, "Options");
  if (!an_get_bool(env, x, &out->translate)) return ng_leaf(fail, "Bool", "translate");
  if (!an_get_field(env, t, WHISPER_TYPE_Options, 2, &x)) return ng_expected(fail, "Options");
  if (!an_get_int(env, x, &out->beam_size)) return ng_leaf(fail, "Int", "beam_size");
  if (!an_get_field(env, t, WHISPER_TYPE_Options, 3, &x)) return ng_expected(fail, "Options");
  if (!an_get_bool(env, x, &out->token_timestamps)) return ng_leaf(fail, "Bool", "token_timestamps");
  if (!an_get_field(env, t, WHISPER_TYPE_Options, 4, &x)) return ng_expected(fail, "Options");
  if (!an_get_int(env, x, &out->max_len)) return ng_leaf(fail, "Int", "max_len");
  if (!an_get_field(env, t, WHISPER_TYPE_Options, 5, &x)) return ng_expected(fail, "Options");
  if (!an_get_bool(env, x, &out->language_probabilities)) return ng_leaf(fail, "Bool", "language_probabilities");
  return true;
}

bool whisper_Options_get(an_env *env, an_term t, whisper_Options *out) {
  ng_fail fail;
  return whisper_native_get_Options(env, t, out, &fail) || ng_raise(env, "", "Options", &fail);
}

an_term whisper_Segment_make(an_env *env, const whisper_Segment *value) {
  an_term fields[3];
  fields[0] = an_int(env, value->start_ms);
  fields[1] = an_int(env, value->end_ms);
  fields[2] = an_string(env, value->text.data, value->text.len);
  return an_struct(env, WHISPER_TYPE_Segment, fields);
}

an_term whisper_Transcript_make(an_env *env, const whisper_Transcript *value) {
  an_term fields[3];
  fields[0] = an_string(env, value->language.data, value->language.len);
  fields[1] = value->segments;
  fields[2] = value->detection;
  return an_struct(env, WHISPER_TYPE_Transcript, fields);
}

an_term whisper_SegmentDetail_make(an_env *env, const whisper_SegmentDetail *value) {
  an_term fields[6];
  fields[0] = an_int(env, value->start_ms);
  fields[1] = an_int(env, value->end_ms);
  fields[2] = an_string(env, value->text.data, value->text.len);
  fields[3] = value->tokens;
  fields[4] = an_float(env, value->avg_logprob);
  fields[5] = an_float(env, value->no_speech_probability);
  return an_struct(env, WHISPER_TYPE_SegmentDetail, fields);
}

an_term whisper_Detection_make(an_env *env, const whisper_Detection *value) {
  an_term fields[3];
  fields[0] = an_string(env, value->language.data, value->language.len);
  fields[1] = an_float(env, value->probability);
  fields[2] = value->probabilities;
  return an_struct(env, WHISPER_TYPE_Detection, fields);
}

an_term whisper_Token_make(an_env *env, const whisper_Token *value) {
  an_term fields[6];
  fields[0] = value->ids;
  fields[1] = an_string(env, value->text.data, value->text.len);
  fields[2] = an_float(env, value->probability);
  fields[3] = an_float(env, value->log_probability);
  fields[4] = an_int(env, value->start_ms);
  fields[5] = an_int(env, value->end_ms);
  return an_struct(env, WHISPER_TYPE_Token, fields);
}

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

void whisper_list_Segment_begin(an_env *env, whisper_list_Segment_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void whisper_list_Segment_push(an_env *env, whisper_list_Segment_builder *b, whisper_Segment value) {
  an_list_push(env, &b->builder, whisper_Segment_make(env, &value));
}

an_term whisper_list_Segment_end(an_env *env, whisper_list_Segment_builder *b) { return an_list_end(env, &b->builder); }

an_term whisper_result_list_Segment_ok(an_env *env, an_term value) {
  return an_ok(env, value);
}

an_term whisper_result_Transcript_ok(an_env *env, whisper_Transcript value) {
  return an_ok(env, whisper_Transcript_make(env, &value));
}

an_term whisper_optional_String_some(an_env *env, ng_str value) {
  return an_some(env, an_string(env, value.data, value.len));
}

void whisper_list_SegmentDetail_begin(an_env *env, whisper_list_SegmentDetail_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void whisper_list_SegmentDetail_push(an_env *env, whisper_list_SegmentDetail_builder *b, whisper_SegmentDetail value) {
  an_list_push(env, &b->builder, whisper_SegmentDetail_make(env, &value));
}

an_term whisper_list_SegmentDetail_end(an_env *env, whisper_list_SegmentDetail_builder *b) { return an_list_end(env, &b->builder); }

an_term whisper_optional_Detection_some(an_env *env, whisper_Detection value) {
  return an_some(env, whisper_Detection_make(env, &value));
}

void whisper_list_Token_begin(an_env *env, whisper_list_Token_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void whisper_list_Token_push(an_env *env, whisper_list_Token_builder *b, whisper_Token value) {
  an_list_push(env, &b->builder, whisper_Token_make(env, &value));
}

an_term whisper_list_Token_end(an_env *env, whisper_list_Token_builder *b) { return an_list_end(env, &b->builder); }

void whisper_map_String_Float_begin(an_env *env, whisper_map_String_Float_builder *b) { b->map = an_map_from(env, NULL, NULL, 0); }

void whisper_map_String_Float_put(an_env *env, whisper_map_String_Float_builder *b, ng_str key, double value) {
  b->map = an_map_put(env, b->map, an_string(env, key.data, key.len), an_float(env, value));
}

an_term whisper_map_String_Float_end(an_env *env, whisper_map_String_Float_builder *b) {
  (void)env;
  return b->map;
}

void whisper_list_Int_begin(an_env *env, whisper_list_Int_builder *b, size_t hint) { an_list_begin(env, &b->builder, hint); }

void whisper_list_Int_push(an_env *env, whisper_list_Int_builder *b, int64_t value) {
  an_list_push(env, &b->builder, an_int(env, value));
}

an_term whisper_list_Int_end(an_env *env, whisper_list_Int_builder *b) { return an_list_end(env, &b->builder); }

static an_term whisper_native_load_model(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_str a0;
  if (!an_get_string(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  return whisper_load_model(env, a0);
}

static an_term whisper_native_transcribe_with(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct whisper_Model_state *a0 = whisper_Model_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  whisper_list_Float a1;
  if (!whisper_native_open_list_Float(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Float]", 1);
  a1.argument = 2;
  whisper_Options a2;
  if (!whisper_native_get_Options(env, argv[2], &a2, &fail)) return ng_fail_argument(env, &fail, "Options", 2);
  return whisper_transcribe_with(env, a0, a1, a2);
}

static an_term whisper_native_transcribe_detailed(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_fail fail;
  struct whisper_Model_state *a0 = whisper_Model_get(env, argv[0]);
  if (!a0) return an_badarg(env, 0);
  whisper_list_Float a1;
  if (!whisper_native_open_list_Float(env, argv[1], &a1, &fail))
    return ng_fail_argument(env, &fail, "List[Float]", 1);
  a1.argument = 2;
  whisper_Options a2;
  if (!whisper_native_get_Options(env, argv[2], &a2, &fail)) return ng_fail_argument(env, &fail, "Options", 2);
  return whisper_transcribe_detailed(env, a0, a1, a2);
}

static an_term whisper_native_language_name(an_env *env, int argc, const an_term *argv) {
  (void)argc;
  ng_str a0;
  if (!an_get_string(env, argv[0], &a0.data, &a0.len)) return an_badarg(env, 0);
  return whisper_language_name(env, a0);
}

static const an_function whisper_native_functions[] = {
  {"load_model", "(String) -> Result[Model]", whisper_native_load_model, AN_DIRTY},
  {"transcribe_with", "(Model, List[Float], Options) -> Result[List[Segment]]", whisper_native_transcribe_with, AN_DIRTY},
  {"transcribe_detailed", "(Model, List[Float], Options) -> Result[Transcript]", whisper_native_transcribe_detailed, AN_DIRTY},
  {"language_name", "(String) -> Optional[String]", whisper_native_language_name, AN_SHORT},
};

static const an_type whisper_native_types[] = {
  {"Options", AN_STRUCT, "language: String, translate: Bool, beam_size: Int, token_timestamps: Bool, max_len: Int, language_probabilities: Bool"},
  {"Segment", AN_STRUCT, "start_ms: Int, end_ms: Int, text: String"},
  {"Transcript", AN_STRUCT, "language: String, segments: List[SegmentDetail], detection: Optional[Detection]"},
  {"SegmentDetail", AN_STRUCT, "start_ms: Int, end_ms: Int, text: String, tokens: List[Token], avg_logprob: Float, no_speech_probability: Float"},
  {"Detection", AN_STRUCT, "language: String, probability: Float, probabilities: Map[String, Float]"},
  {"Token", AN_STRUCT, "ids: List[Int], text: String, probability: Float, log_probability: Float, start_ms: Int, end_ms: Int"},
};

/* Ends the module's one .c file: the tables and agentica_native_init. LOAD is NULL or a const char *(*)(void)
   that runs once at the first call, and refuses the module with the message it returns. */
#define WHISPER_NATIVE_MODULE(LOAD) \
  static const an_resource_type whisper_native_resources[] = { \
    {"Model", sizeof(struct whisper_Model_state), WHISPER_Model_FREE}, \
  }; \
  AN_MODULE_TABLES("whisper", whisper_native_functions, 4, whisper_native_types, 6, whisper_native_resources, 1, LOAD)
#endif
