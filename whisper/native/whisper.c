/* The thin C layer of whisper.agc, on the header that native-gen makes from its
   `extern module` block. It answers error until whisper.cpp is linked (Fase W,
   Via A): loading and transcribing need the C++ engine, which the scaffold
   does not vendor. Do not edit the generated header: change the .agc instead,
   and make the header again (see its header comment). */
#define WHISPER_NATIVE_IMPLEMENTATION
#include "whisper_native.h"

/* The state of a loaded model; the engine owns it once linked. C11 has no
   empty struct, so the stub keeps one member. */
struct whisper_Model_state {
  int unused;
};
#define WHISPER_Model_FREE NULL

an_term whisper_load_model(an_env *env, ng_str path) {
  (void)path;
  static const char message[] = "whisper is not linked yet: no model can load";
  return an_error(env, message, sizeof message - 1);
}

an_term whisper_transcribe(an_env *env, struct whisper_Model_state *model, whisper_list_Float pcm) {
  (void)model;
  (void)pcm;
  static const char message[] = "whisper is not linked yet: nothing can transcribe";
  return an_error(env, message, sizeof message - 1);
}

WHISPER_NATIVE_MODULE(NULL)
