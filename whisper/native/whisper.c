/* The thin C layer of whisper.agc, on the header that native-gen makes from its
   `extern module` block. It links the pinned whisper.cpp engine (CPU-only, no
   GPU): `tools/build-engine.sh` builds its static libraries, and the build
   table of `agentica.toml` links them (`C_INCLUDE_PATH` and `LIBRARY_PATH`
   point at the engine; see `README.md`). Do not edit the generated header:
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

/* The engine enters only as the pinned submodule: never copied into native/
   (a fetched copy of this package has no third_party/, so it uses the
   prebuilt libraries instead). The path is relative to this file, so an
   in-place checkout finds it; ggml/*.h come from C_INCLUDE_PATH. */
#include "../../third_party/whisper.cpp/include/whisper.h"

/* Bounds, so one call cannot grow without limit: a path the loader copies
   once, five minutes of 16 kHz mono audio, a language code (the engine's own
   are two or three letters, "auto" included), and a megabyte of text out. */
#define WHISPER_PATH_MAX 4096
#define WHISPER_MAX_PCM_SAMPLES (16000 * 60 * 5)
#define WHISPER_LANG_MAX 32
#define WHISPER_MAX_TEXT_BYTES (1 << 20)

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

  struct whisper_context_params cparams = whisper_context_default_params();
  cparams.use_gpu = false;
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

an_term whisper_transcribe_with(an_env *env, struct whisper_Model_state *model, whisper_list_Float pcm,
                                whisper_Options options) {
  if (model == NULL || !model->live || model->ctx == NULL) {
    static const char message[] = "transcribe_with needs a loaded model";
    return an_error(env, message, sizeof message - 1);
  }

  if (pcm.len == 0) {
    static const char message[] = "transcribe_with needs at least one sample at 16000 Hz";
    return an_error(env, message, sizeof message - 1);
  }

  if (pcm.len > WHISPER_MAX_PCM_SAMPLES) {
    static const char message[] = "transcribe_with takes five minutes of audio at most (4800000 samples at 16000 Hz)";
    return an_error(env, message, sizeof message - 1);
  }

  if (options.language.len > WHISPER_LANG_MAX) {
    static const char message[] = "transcribe_with takes a language code of 32 bytes at most";
    return an_error(env, message, sizeof message - 1);
  }

  float * samples = (float *)malloc(pcm.len * sizeof(float));

  if (samples == NULL) {
    static const char message[] = "transcribe_with ran out of memory for the audio";
    return an_error(env, message, sizeof message - 1);
  }

  double x = 0.0;
  size_t n_samples = 0;
  int more = 0;

  while ((more = whisper_list_Float_next(env, &pcm, &x)) == 1) {
    /* The cast below can round a finite double to infinite float (1.0e300
       and kin), so the range is checked on the double: only values the
       float keeps reach whisper_full, per the documented contract. */
    if (!isfinite(x) || fabs(x) > (double)FLT_MAX) {
      free(samples);
      static const char message[] = "transcribe_with needs finite samples";
      return an_error(env, message, sizeof message - 1);
    }

    samples[n_samples++] = (float)x;
  }

  if (more < 0) {
    free(samples);
    return an_void(env);
  }

  struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
  wparams.print_progress = false;
  wparams.print_special = false;
  wparams.print_realtime = false;
  wparams.print_timestamps = false;
  /* The segment path times what it emits, so the timestamps stay on (the
     engine default) and whisper_full_get_segment_t0/t1 below report them. */
  wparams.no_timestamps = false;
  wparams.translate = options.translate;

  /* The engine holds `language` as a C string, while a String argument is
     UTF-8 with no NUL at its end: the copy gives it one. "" keeps the
     library default ("en"), and "auto" or a code such as "es" goes through
     as it is (the engine reads "auto" as detection). Greedy sampling and the
     library thread count stay: this binding adds no other option. */
  char language[WHISPER_LANG_MAX + 1];

  if (options.language.len > 0) {
    memcpy(language, options.language.data, options.language.len);
    language[options.language.len] = '\0';
    wparams.language = language;
  }

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
    static const char failed[] = "the engine could not transcribe the audio";
    const char * message = rc == 2 ? overflowing : failed;
    return an_error(env, message, strlen(message));
  }

  return whisper_result_list_Segment_ok(env, segments);
}

WHISPER_NATIVE_MODULE(NULL)
