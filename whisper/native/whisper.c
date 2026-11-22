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
   once, five minutes of 16 kHz mono audio, and a megabyte of text out. */
#define WHISPER_PATH_MAX 4096
#define WHISPER_MAX_PCM_SAMPLES (16000 * 60 * 5)
#define WHISPER_MAX_TEXT_BYTES (1 << 20)

/* The state of a loaded model. The mutex serializes inference on the context:
   whisper_full is not thread-safe on one context, while two tasks may call
   transcribe with the same Model at once (each dirty call runs on the compute
   pool). FREE never runs while a call holds the resource, so destroying the
   mutex there is safe. `live` tells FREE what to undo: the state starts
   zeroed, and load_model sets it once the mutex exists. */
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

an_term whisper_transcribe(an_env *env, struct whisper_Model_state *model, whisper_list_Float pcm) {
  if (model == NULL || !model->live || model->ctx == NULL) {
    static const char message[] = "transcribe needs a loaded model";
    return an_error(env, message, sizeof message - 1);
  }

  if (pcm.len == 0) {
    static const char message[] = "transcribe needs at least one sample at 16000 Hz";
    return an_error(env, message, sizeof message - 1);
  }

  if (pcm.len > WHISPER_MAX_PCM_SAMPLES) {
    static const char message[] = "transcribe takes five minutes of audio at most (4800000 samples at 16000 Hz)";
    return an_error(env, message, sizeof message - 1);
  }

  float * samples = (float *)malloc(pcm.len * sizeof(float));

  if (samples == NULL) {
    static const char message[] = "transcribe ran out of memory for the audio";
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
      static const char message[] = "transcribe needs finite samples";
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
  wparams.no_timestamps = true;

  /* Greedy decoding is deterministic; the library default language ("en")
     and thread count stay, so this binding adds no options of its own. */
  if (pthread_mutex_lock(&model->mutex) != 0) {
    free(samples);
    static const char message[] = "transcribe cannot lock the model";
    return an_error(env, message, sizeof message - 1);
  }

  int rc = whisper_full(model->ctx, wparams, samples, (int)n_samples);

  char * text = NULL;
  size_t text_len = 0;
  size_t text_cap = 0;

  if (rc == 0) {
    int n_segments = whisper_full_n_segments(model->ctx);

    for (int i = 0; i < n_segments && rc == 0; i++) {
      const char * segment = whisper_full_get_segment_text(model->ctx, i);
      size_t piece = strlen(segment);

      if (text_len + piece > WHISPER_MAX_TEXT_BYTES) {
        rc = 2;
        break;
      }

      if (text_len + piece + 1 > text_cap) {
        size_t cap = text_cap == 0 ? 1024 : text_cap * 2;

        while (cap < text_len + piece + 1) {
          cap *= 2;
        }

        char * grown = (char *)realloc(text, cap);

        if (grown == NULL) {
          rc = 3;
          break;
        }

        text = grown;
        text_cap = cap;
      }

      memcpy(text + text_len, segment, piece);
      text_len += piece;
    }
  }

  pthread_mutex_unlock(&model->mutex);
  free(samples);

  if (rc != 0) {
    free(text);
    static const char overflowing[] = "transcribe ran past a megabyte of text";
    static const char failed[] = "the engine could not transcribe the audio";
    const char * message = rc == 2 ? overflowing : failed;
    return an_error(env, message, strlen(message));
  }

  if (text == NULL) {
    static const char message[] = "";
    return whisper_result_String_ok(env, (ng_str){message, 0});
  }

  an_term result = whisper_result_String_ok(env, (ng_str){text, text_len});
  free(text);

  return result;
}

WHISPER_NATIVE_MODULE(NULL)
