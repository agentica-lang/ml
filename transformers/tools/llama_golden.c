/* The golden of `qwen_check.agc` made by llama.cpp from a GGUF file: for each line of IDS_FILE (token ids, separated by
   spaces) the logits of the last position, and the ids of the five largest logits of every position, as JSON.
   Manual, not built by CI: it needs a llama.cpp checkout and its build (libllama).

     cc -O2 -o llama_golden tools/llama_golden.c -I$LLAMA/include -I$LLAMA/ggml/include -L$BUILD/bin -lllama -lggml \
        -lggml-base -Wl,-rpath,$BUILD/bin
     ./llama_golden MODEL.gguf IDS_FILE > GOLDEN.json
     WEIGHTS=q8_0 QUANTIZED=1 agentica tools/qwen_check.agc MODEL.gguf GOLDEN.json cpu

   It runs on the CPU, with the same quantized products (Q8_0, Q4_K, Q6_K) as ggml's CPU backend in `tensor/`. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

#define MAX_IDS 4096

static void top5(const float *logits, int n, int out[5]) {
  int i, k, j;

  for (k = 0; k < 5; k++) out[k] = -1;
  for (i = 0; i < n; i++) {
    for (k = 0; k < 5; k++) {
      if (out[k] < 0 || logits[i] > logits[out[k]]) {
        for (j = 4; j > k; j--) out[j] = out[j - 1];
        out[k] = i;
        break;
      }
    }
  }
}

int main(int argc, char **argv) {
  struct llama_model_params mp = llama_model_default_params();
  struct llama_context_params cp = llama_context_default_params();
  struct llama_model *model;
  struct llama_context *ctx;
  const struct llama_vocab *vocab;
  FILE *in;
  char *line = (char *)malloc(MAX_IDS * 12);
  int first = 1, n_vocab;

  if (argc != 3 || line == NULL) {
    fprintf(stderr, "usage: llama_golden MODEL.gguf IDS_FILE\n");
    return 2;
  }
  in = fopen(argv[2], "r");
  if (in == NULL) {
    fprintf(stderr, "llama_golden: cannot open %s\n", argv[2]);
    return 1;
  }
  llama_backend_init();
  mp.n_gpu_layers = 0;
  model = llama_model_load_from_file(argv[1], mp);
  if (model == NULL) {
    fprintf(stderr, "llama_golden: cannot load %s\n", argv[1]);
    return 1;
  }
  vocab = llama_model_get_vocab(model);
  n_vocab = llama_vocab_n_tokens(vocab);
  printf("{\"cases\": [");
  while (fgets(line, MAX_IDS * 12, in) != NULL) {
    llama_token ids[MAX_IDS];
    struct llama_batch batch;
    int n = 0, i, k;
    char *p = line, *end;
    size_t len = strlen(line);

    if (len == MAX_IDS * 12 - 1 && line[len - 1] != '\n') {
      fprintf(stderr, "llama_golden: a line is longer than %d bytes\n", MAX_IDS * 12 - 1);
      return 1;
    }
    while (n < MAX_IDS && (ids[n] = (llama_token)strtol(p, &end, 10), end != p)) {
      n++;
      p = end;
    }
    if (n == MAX_IDS && (strtol(p, &end, 10), end != p)) {
      fprintf(stderr, "llama_golden: a case has more than %d ids\n", MAX_IDS);
      return 1;
    }
    if (n == 0) continue;
    cp.n_ctx = (uint32_t)n;
    cp.n_batch = (uint32_t)n;
    cp.n_ubatch = (uint32_t)n;
    ctx = llama_init_from_model(model, cp);
    if (ctx == NULL) {
      fprintf(stderr, "llama_golden: no context for %d tokens\n", n);
      return 1;
    }
    batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (i = 0; i < n; i++) {
      batch.token[i] = ids[i];
      batch.pos[i] = i;
      batch.n_seq_id[i] = 1;
      batch.seq_id[i][0] = 0;
      batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch) != 0) {
      fprintf(stderr, "llama_golden: decode failed\n");
      return 1;
    }
    printf("%s{\"ids\": [", first ? "" : ", ");
    first = 0;
    for (i = 0; i < n; i++) printf("%s%d", i ? ", " : "", (int)ids[i]);
    printf("], \"last\": [");
    for (k = 0; k < n_vocab; k++) printf("%s%.9g", k ? ", " : "", (double)llama_get_logits_ith(ctx, n - 1)[k]);
    printf("], \"top5\": [");
    for (i = 0; i < n; i++) {
      int best[5];

      top5(llama_get_logits_ith(ctx, i), n_vocab, best);
      printf("%s[%d, %d, %d, %d, %d]", i ? ", " : "", best[0], best[1], best[2], best[3], best[4]);
    }
    printf("]}");
    llama_batch_free(batch);
    llama_free(ctx);
  }
  printf("]}\n");
  llama_model_free(model);
  llama_backend_free();
  fclose(in);
  free(line);
  return 0;
}
