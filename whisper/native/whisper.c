/* The thin C layer of whisper.agc, on the header that native-gen makes from its
   `extern module` block. It links the pinned whisper.cpp engine, built for the
   CPU or with Vulkan: `tools/build-engine.sh` builds its static libraries, and
   the build table of `agentica.toml` links them (`C_INCLUDE_PATH` and
   `LIBRARY_PATH` point at the engine; see `README.md`). Do not edit the generated header:
   change the .agc instead, and make the header again (see its header
   comment). */
#define WHISPER_NATIVE_IMPLEMENTATION
#include "whisper_native.h"

#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The engine enters only as the pinned submodule, never copied into native/:
   whisper.h and ggml's headers come from C_INCLUDE_PATH, so a fetched copy of
   this package, which has no third_party/, builds against the engine that
   path names. */
#include <whisper.h>

/* Bounds, so one call cannot grow without limit: a path the loader copies
   once, five minutes of 16 kHz mono audio, a language code (the engine's own
   are two or three letters, "auto" included), and a megabyte of text out. */
#define WHISPER_PATH_MAX 4096
#define WHISPER_MAX_PCM_SAMPLES (16000 * 60 * 5)
#define WHISPER_LANG_MAX 32
#define WHISPER_MAX_TEXT_BYTES (1 << 20)
#define WHISPER_MAX_BEAMS 8
#define WHISPER_MAX_LEN 100000

/* The state of a loaded model. The mutex serializes inference on the context:
   whisper_full is not thread-safe on one context, while two tasks may call
   transcribe_with with the same Model at once (each dirty call runs on the
   compute pool). FREE never runs while a call holds the resource, so
   destroying the mutex there is safe. `live` tells FREE what to undo: the
   state starts zeroed, and load_model sets it once the mutex exists. */
struct whisper_Model_state {
  struct whisper_context * ctx;
  pthread_mutex_t mutex;
  bool live;
};

static void whisper_Model_free(void * state) {
  struct whisper_Model_state * self = (struct whisper_Model_state *)state;

  if (!self->live) {
    return;
  }

  if (self->ctx != NULL) {
    whisper_free(self->ctx);
    self->ctx = NULL;
  }

  pthread_mutex_destroy(&self->mutex);
  self->live = false;
}
#define WHISPER_Model_FREE whisper_Model_free

an_term whisper_load_model(an_env *env, ng_str path) {
  an_term out = an_void(env);
  struct whisper_Model_state * self = whisper_Model_new(env, &out);

  if (self == NULL) {
    return an_void(env);
  }

  if (path.len > WHISPER_PATH_MAX) {
    static const char message[] = "load_model takes a path of 4096 bytes at most";
    return an_error(env, message, sizeof message - 1);
  }

  if (pthread_mutex_init(&self->mutex, NULL) != 0) {
    static const char message[] = "load_model cannot start the model: no mutex";
    return an_error(env, message, sizeof message - 1);
  }

  self->live = true;

  char * copy = (char *)malloc(path.len + 1);

  if (copy == NULL) {
    static const char message[] = "load_model ran out of memory for the path";
    return an_error(env, message, sizeof message - 1);
  }

  memcpy(copy, path.data, path.len);
  copy[path.len] = '\0';

  /* use_gpu takes the GPU backend the engine was built with (Vulkan), and
     the CPU when it has none. */
  struct whisper_context_params cparams = whisper_context_default_params();
  cparams.use_gpu = true;
  self->ctx = whisper_init_from_file_with_params(copy, cparams);
  free(copy);

  if (self->ctx == NULL) {
    char message[512];
    int n = snprintf(message, sizeof message,
                     "cannot load the model at %.*s: whisper_init_from_file_with_params gave NULL",
                     (int)(path.len < 400 ? path.len : 400), path.data);

    if (n < 0) {
      static const char fallback[] = "cannot load the model: whisper_init_from_file_with_params gave NULL";
      return an_error(env, fallback, sizeof fallback - 1);
    }

    return an_error(env, message, (size_t)(n < (int)sizeof message ? n : (int)sizeof message - 1));
  }

  return whisper_result_Model_ok(env, out);
}

/* Whether C is whitespace the segments are trimmed of. Only ASCII: the engine
   pads a segment with spaces, and every other byte of its UTF-8 text belongs
   to it. */
static bool whisper_is_space(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

/* The checks every decode shares, before any work: the message of the first
   one that fails, or NULL. */
static const char * whisper_check_call(const struct whisper_Model_state * model, const whisper_list_Float * pcm,
                                       const whisper_Options * options) {
  if (model == NULL || !model->live || model->ctx == NULL) {
    return "transcribe_with needs a loaded model";
  }

  if (pcm->len == 0) {
    return "transcribe_with needs at least one sample at 16000 Hz";
  }

  if (pcm->len > WHISPER_MAX_PCM_SAMPLES) {
    return "transcribe_with takes five minutes of audio at most (4800000 samples at 16000 Hz)";
  }

  if (options->language.len > WHISPER_LANG_MAX) {
    return "transcribe_with takes a language code of 32 bytes at most";
  }

  if (options->beam_size < 0 || options->beam_size > WHISPER_MAX_BEAMS) {
    return "transcribe_with takes a beam_size from 0 to 8";
  }

  if (options->max_len < 0 || options->max_len > WHISPER_MAX_LEN) {
    return "transcribe_with takes a max_len from 0 to 100000";
  }

  return NULL;
}

/* Copies the samples of PCM into *OUT (malloc'd; the caller frees it). Gives
   NULL when done, a message for an error, and sets *FAILED instead when the
   list did not decode (the call has then failed already). */
static const char * whisper_take_samples(an_env * env, whisper_list_Float * pcm, float ** out, size_t * count,
                                         bool * failed) {
  float * samples = (float *)malloc(pcm->len * sizeof(float));

  if (samples == NULL) {
    return "transcribe_with ran out of memory for the audio";
  }

  double x = 0.0;
  size_t n_samples = 0;
  int more = 0;

  while ((more = whisper_list_Float_next(env, pcm, &x)) == 1) {
    /* The cast below can round a finite double to infinite float (1.0e300
       and kin), so the range is checked on the double: only values the
       float keeps reach whisper_full, per the documented contract. */
    if (!isfinite(x) || fabs(x) > (double)FLT_MAX) {
      free(samples);
      return "transcribe_with needs finite samples";
    }

    samples[n_samples++] = (float)x;
  }

  if (more < 0) {
    free(samples);
    *failed = true;
    return NULL;
  }

  *out = samples;
  *count = n_samples;
  return NULL;
}

/* The engine parameters OPTIONS asks for. LANGUAGE is the caller's buffer of
   WHISPER_LANG_MAX + 1 bytes: the engine holds `language` as a C string,
   while a String argument is UTF-8 with no NUL at its end, so the copy gives
   it one. "" keeps the library default ("en"), and "auto" or a code such as
   "es" goes through as it is (the engine reads "auto" as detection). */
static struct whisper_full_params whisper_params_for(const whisper_Options * options, char * language) {
  bool beam = options->beam_size > 1;
  struct whisper_full_params wparams =
      whisper_full_default_params(beam ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);

  if (beam) {
    wparams.beam_search.beam_size = (int)options->beam_size;
  }

  wparams.print_progress = false;
  wparams.print_special = false;
  wparams.print_realtime = false;
  wparams.print_timestamps = false;
  /* The segment path times what it emits, so the timestamps stay on (the
     engine default) and whisper_full_get_segment_t0/t1 report them. */
  wparams.no_timestamps = false;
  wparams.translate = options->translate;
  wparams.token_timestamps = options->token_timestamps;
  wparams.max_len = (int)options->max_len;

  if (options->language.len > 0) {
    memcpy(language, options->language.data, options->language.len);
    language[options->language.len] = '\0';
    wparams.language = language;
  }

  return wparams;
}

an_term whisper_transcribe_with(an_env *env, struct whisper_Model_state *model, whisper_list_Float pcm,
                                whisper_Options options) {
  const char * problem = whisper_check_call(model, &pcm, &options);

  if (problem != NULL) {
    return an_error(env, problem, strlen(problem));
  }

  float * samples = NULL;
  size_t n_samples = 0;
  bool failed = false;
  problem = whisper_take_samples(env, &pcm, &samples, &n_samples, &failed);

  if (failed) {
    return an_void(env);
  }

  if (problem != NULL) {
    return an_error(env, problem, strlen(problem));
  }

  char language[WHISPER_LANG_MAX + 1];
  struct whisper_full_params wparams = whisper_params_for(&options, language);

  if (pthread_mutex_lock(&model->mutex) != 0) {
    free(samples);
    static const char message[] = "transcribe_with cannot lock the model";
    return an_error(env, message, sizeof message - 1);
  }

  int rc = whisper_full(model->ctx, wparams, samples, (int)n_samples);
  an_term segments = an_void(env);

  if (rc == 0) {
    int n_segments = whisper_full_n_segments(model->ctx);
    whisper_list_Segment_builder out;
    size_t text_len = 0;

    whisper_list_Segment_begin(env, &out, (size_t)(n_segments > 0 ? n_segments : 0));

    for (int i = 0; i < n_segments; i++) {
      const char * raw = whisper_full_get_segment_text(model->ctx, i);
      size_t raw_len = strlen(raw);
      size_t first = 0;
      size_t last = raw_len;

      while (first < last && whisper_is_space(raw[first])) {
        first++;
      }

      while (last > first && whisper_is_space(raw[last - 1])) {
        last--;
      }

      size_t piece = last - first;

      if (text_len + piece > WHISPER_MAX_TEXT_BYTES) {
        rc = 2;
        break;
      }

      /* t0 and t1 are hundredths of a second; ms is what Segment promises.
         an_string copies the bytes, so the engine's own text can go once the
         lock is released. */
      whisper_Segment segment;
      segment.start_ms = whisper_full_get_segment_t0(model->ctx, i) * 10;
      segment.end_ms = whisper_full_get_segment_t1(model->ctx, i) * 10;
      segment.text = (ng_str){raw + first, piece};
      whisper_list_Segment_push(env, &out, segment);
      text_len += piece;
    }

    segments = whisper_list_Segment_end(env, &out);
  }

  pthread_mutex_unlock(&model->mutex);
  free(samples);

  if (rc != 0) {
    static const char overflowing[] = "transcribe_with ran past a megabyte of text";
    static const char failed_decode[] = "the engine could not transcribe the audio";
    const char * message = rc == 2 ? overflowing : failed_decode;
    return an_error(env, message, strlen(message));
  }

  return whisper_result_list_Segment_ok(env, segments);
}

/* A growable byte buffer for the text the detailed call cleans or joins. */
struct whisper_text {
  char * data;
  size_t len;
  size_t cap;
};

static bool whisper_text_reserve(struct whisper_text * text, size_t more) {
  if (text->len + more <= text->cap) {
    return true;
  }

  size_t cap = text->cap > 0 ? text->cap : 256;

  while (cap < text->len + more) {
    cap *= 2;
  }

  char * grown = (char *)realloc(text->data, cap);

  if (grown == NULL) {
    return false;
  }

  text->data = grown;
  text->cap = cap;
  return true;
}

/* The length of the well-formed UTF-8 sequence at IN (LEN bytes left), or 0
   when it is not one: an overlong form, a surrogate, past U+10FFFF or cut. */
static size_t whisper_utf8_sequence(const unsigned char * in, size_t len) {
  unsigned char c = in[0];

  if (c < 0x80) {
    return 1;
  }

  size_t need = 0;
  unsigned char low = 0x80;
  unsigned char high = 0xBF;

  if (c >= 0xC2 && c <= 0xDF) {
    need = 2;
  } else if (c >= 0xE0 && c <= 0xEF) {
    need = 3;
    low = c == 0xE0 ? 0xA0 : 0x80;
    high = c == 0xED ? 0x9F : 0xBF;
  } else if (c >= 0xF0 && c <= 0xF4) {
    need = 4;
    low = c == 0xF0 ? 0x90 : 0x80;
    high = c == 0xF4 ? 0x8F : 0xBF;
  } else {
    return 0;
  }

  if (len < need || in[1] < low || in[1] > high) {
    return 0;
  }

  for (size_t k = 2; k < need; k++) {
    if (in[k] < 0x80 || in[k] > 0xBF) {
      return 0;
    }
  }

  return need;
}

/* Replaces TEXT with its bytes as valid UTF-8: each byte that starts no
   well-formed sequence becomes U+FFFD, as whisper.cpp's server writes it. The
   engine cuts text on token boundaries, which may fall inside a character.
   SCRATCH holds the copy; false when it runs out of memory. */
static bool whisper_utf8_clean(struct whisper_text * scratch, const char * in, size_t len, ng_str * out) {
  scratch->len = 0;

  for (size_t at = 0; at < len;) {
    size_t n = whisper_utf8_sequence((const unsigned char *)in + at, len - at);

    if (!whisper_text_reserve(scratch, n > 0 ? n : 3)) {
      return false;
    }

    if (n > 0) {
      memcpy(scratch->data + scratch->len, in + at, n);
      scratch->len += n;
      at += n;
    } else {
      memcpy(scratch->data + scratch->len, "\xEF\xBF\xBD", 3);
      scratch->len += 3;
      at += 1;
    }
  }

  *out = (ng_str){scratch->data, scratch->len};
  return true;
}

/* How many bytes the character that TEXT ends in still lacks (0 when it is
   whole or not a UTF-8 lead at all): whisper.cpp's utf8_trailing_bytes_needed,
   which decides when the server joins a token with the next. */
static int whisper_utf8_missing(const char * text, size_t len) {
  size_t i = len;

  while (i > 0 && ((unsigned char)text[i - 1] & 0xC0) == 0x80) {
    i--;
  }

  if (i == 0) {
    return 0;
  }

  unsigned char c = (unsigned char)text[i - 1];
  int expected = 0;

  if ((c & 0x80) == 0x00) {
    expected = 1;
  } else if ((c & 0xE0) == 0xC0) {
    expected = 2;
  } else if ((c & 0xF0) == 0xE0) {
    expected = 3;
  } else if ((c & 0xF8) == 0xF0) {
    expected = 4;
  } else {
    return 0;
  }

  int have = (int)(len - (i - 1));
  return have < expected ? expected - have : 0;
}

/* The segments of the decode on CTX, each with its tokens, as a List term.
   Gives 0, 1 when the text passed WHISPER_MAX_TEXT_BYTES and 3 when memory
   ran out. */
static int whisper_segment_details(an_env * env, struct whisper_context * ctx, an_term * out) {
  int n_segments = whisper_full_n_segments(ctx);
  whisper_token eot = whisper_token_eot(ctx);
  struct whisper_text joined = {0};
  struct whisper_text clean = {0};
  whisper_list_SegmentDetail_builder segments;
  size_t text_len = 0;
  int rc = 0;

  whisper_list_SegmentDetail_begin(env, &segments, (size_t)(n_segments > 0 ? n_segments : 0));

  for (int i = 0; i < n_segments && rc == 0; i++) {
    int n_tokens = whisper_full_n_tokens(ctx, i);
    whisper_list_Token_builder tokens;
    double total_logprob = 0.0;

    whisper_list_Token_begin(env, &tokens, (size_t)(n_tokens > 0 ? n_tokens : 0));

    for (int j = 0; j < n_tokens && rc == 0; j++) {
      whisper_token_data first = whisper_full_get_token_data(ctx, i, j);

      if (first.id >= eot) {
        continue;
      }

      whisper_list_Int_builder ids;
      const char * piece = whisper_full_get_token_text(ctx, i, j);
      size_t piece_len = strlen(piece);
      int64_t end = first.t1;
      double logprob = first.plog;

      whisper_list_Int_begin(env, &ids, 1);
      whisper_list_Int_push(env, &ids, first.id);
      joined.len = 0;

      if (!whisper_text_reserve(&joined, piece_len)) {
        rc = 3;
        break;
      }

      memcpy(joined.data, piece, piece_len);
      joined.len = piece_len;

      while (j + 1 < n_tokens && whisper_utf8_missing(joined.data, joined.len) > 0) {
        whisper_token_data next = whisper_full_get_token_data(ctx, i, j + 1);

        if (next.id >= eot) {
          break;
        }

        j++;
        piece = whisper_full_get_token_text(ctx, i, j);
        piece_len = strlen(piece);

        if (!whisper_text_reserve(&joined, piece_len)) {
          rc = 3;
          break;
        }

        memcpy(joined.data + joined.len, piece, piece_len);
        joined.len += piece_len;
        whisper_list_Int_push(env, &ids, next.id);

        if (next.t1 > -1) {
          end = next.t1;
        }

        logprob += next.plog;
      }

      text_len += joined.len;

      if (rc == 0 && text_len > WHISPER_MAX_TEXT_BYTES) {
        rc = 1;
      }

      whisper_Token token;

      if (rc == 0 && !whisper_utf8_clean(&clean, joined.data, joined.len, &token.text)) {
        rc = 3;
      }

      if (rc != 0) {
        break;
      }

      token.ids = whisper_list_Int_end(env, &ids);
      token.probability = first.p;
      token.log_probability = logprob;
      token.start_ms = first.t0 * 10;
      token.end_ms = end * 10;
      whisper_list_Token_push(env, &tokens, token);
      total_logprob += logprob;
    }

    if (rc != 0) {
      break;
    }

    const char * raw = whisper_full_get_segment_text(ctx, i);
    size_t raw_len = strlen(raw);
    text_len += raw_len;

    if (text_len > WHISPER_MAX_TEXT_BYTES) {
      rc = 1;
      break;
    }

    whisper_SegmentDetail segment;

    if (!whisper_utf8_clean(&clean, raw, raw_len, &segment.text)) {
      rc = 3;
      break;
    }

    segment.start_ms = whisper_full_get_segment_t0(ctx, i) * 10;
    segment.end_ms = whisper_full_get_segment_t1(ctx, i) * 10;
    segment.tokens = whisper_list_Token_end(env, &tokens);
    segment.avg_logprob = n_tokens > 0 ? total_logprob / n_tokens : 0.0;
    segment.no_speech_probability = whisper_full_get_segment_no_speech_prob(ctx, i);
    whisper_list_SegmentDetail_push(env, &segments, segment);
  }

  free(joined.data);
  free(clean.data);
  *out = whisper_list_SegmentDetail_end(env, &segments);
  return rc;
}

/* The language detection of the first 30 s of the decode on CTX, as an
   Optional[Detection] term in *OUT. Gives 0, or 2 when the engine cannot
   detect and 3 when memory ran out. */
static int whisper_detection(an_env * env, struct whisper_context * ctx, int n_threads, an_term * out) {
  int n_languages = whisper_lang_max_id() + 1;
  float * probabilities = (float *)calloc((size_t)n_languages, sizeof(float));

  if (probabilities == NULL) {
    return 3;
  }

  int best = whisper_lang_auto_detect(ctx, 0, n_threads, probabilities);

  if (best < 0 || best >= n_languages) {
    free(probabilities);
    return 2;
  }

  whisper_map_String_Float_builder map;
  whisper_map_String_Float_begin(env, &map);

  for (int id = 0; id < n_languages; id++) {
    const char * code = whisper_lang_str(id);

    if (code != NULL) {
      whisper_map_String_Float_put(env, &map, (ng_str){code, strlen(code)}, probabilities[id]);
    }
  }

  const char * code = whisper_lang_str(best);
  whisper_Detection detection;
  detection.language = (ng_str){code, strlen(code)};
  detection.probability = probabilities[best];
  detection.probabilities = whisper_map_String_Float_end(env, &map);
  *out = whisper_optional_Detection_some(env, detection);
  free(probabilities);
  return 0;
}

an_term whisper_transcribe_detailed(an_env *env, struct whisper_Model_state *model, whisper_list_Float pcm,
                                    whisper_Options options) {
  const char * problem = whisper_check_call(model, &pcm, &options);

  if (problem != NULL) {
    return an_error(env, problem, strlen(problem));
  }

  float * samples = NULL;
  size_t n_samples = 0;
  bool failed = false;
  problem = whisper_take_samples(env, &pcm, &samples, &n_samples, &failed);

  if (failed) {
    return an_void(env);
  }

  if (problem != NULL) {
    return an_error(env, problem, strlen(problem));
  }

  char language[WHISPER_LANG_MAX + 1];
  struct whisper_full_params wparams = whisper_params_for(&options, language);

  if (pthread_mutex_lock(&model->mutex) != 0) {
    free(samples);
    static const char message[] = "transcribe_with cannot lock the model";
    return an_error(env, message, sizeof message - 1);
  }

  /* 4: the decode itself failed; the other codes are those of the helpers. */
  int rc = whisper_full(model->ctx, wparams, samples, (int)n_samples) == 0 ? 0 : 4;
  whisper_Transcript transcript = {0};

  if (rc == 0) {
    const char * decoded = whisper_lang_str(whisper_full_lang_id(model->ctx));
    transcript.language = decoded != NULL ? (ng_str){decoded, strlen(decoded)} : (ng_str){"", 0};
    rc = whisper_segment_details(env, model->ctx, &transcript.segments);
  }

  transcript.detection = an_none(env);

  if (rc == 0 && options.detect_language) {
    rc = whisper_detection(env, model->ctx, wparams.n_threads, &transcript.detection);
  }

  pthread_mutex_unlock(&model->mutex);
  free(samples);

  switch (rc) {
    case 0:
      return whisper_result_Transcript_ok(env, transcript);
    case 1: {
      static const char message[] = "transcribe_with ran past a megabyte of text";
      return an_error(env, message, sizeof message - 1);
    }
    case 2: {
      static const char message[] = "the engine could not detect the language";
      return an_error(env, message, sizeof message - 1);
    }
    case 3: {
      static const char message[] = "transcribe_with ran out of memory for the text";
      return an_error(env, message, sizeof message - 1);
    }
    default: {
      static const char message[] = "the engine could not transcribe the audio";
      return an_error(env, message, sizeof message - 1);
    }
  }
}

an_term whisper_language_name(an_env *env, ng_str code) {
  char copy[WHISPER_LANG_MAX + 1];

  if (code.len == 0 || code.len > WHISPER_LANG_MAX || memchr(code.data, '\0', code.len) != NULL) {
    return an_none(env);
  }

  memcpy(copy, code.data, code.len);
  copy[code.len] = '\0';
  int id = whisper_lang_id(copy);

  if (id < 0) {
    return an_none(env);
  }

  const char * name = whisper_lang_str_full(id);
  return whisper_optional_String_some(env, (ng_str){name, strlen(name)});
}

WHISPER_NATIVE_MODULE(NULL)
