// Times whisper.cpp the way asr/tools/bench_transcribe.agc times ours: the model is loaded once, one warm-up
// transcription, then N timed ones, each the whole pipeline (log-mel, encoder, greedy decoder) in milliseconds.
//
//   whisper_cpp_bench MODEL.bin CLIP.wav LANGUAGE THREADS USE_GPU FLASH_ATTN [RUNS]
//
// RUNS -1 stops after the load.
//
// It prints tab-separated lines: `load`, `warmup`, `run` (one per run), `text`, `tokens` (the text token ids) and
// `phase NAME MS` (whisper.cpp's own totals over the timed runs, by stage).
// Greedy decoding, no timestamps, no temperature fallback: the settings of `whisper-cli -bs 1 -bo 1 -nt -nf`.
#include "whisper.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// 16-bit PCM WAV, mono, 16 kHz: the samples of its data chunk, scaled to -1..1.
static bool read_wav(const char * path, std::vector<float> & pcm) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    std::vector<unsigned char> b;
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) b.insert(b.end(), buf, buf + n);
    fclose(f);
    size_t at = 12;
    while (at + 8 <= b.size()) {
        uint32_t size = b[at + 4] | b[at + 5] << 8 | b[at + 6] << 16 | (uint32_t) b[at + 7] << 24;
        if (memcmp(&b[at], "data", 4) == 0) {
            for (size_t i = 0; i + 1 < size && at + 8 + i + 1 < b.size(); i += 2) {
                pcm.push_back((int16_t) (b[at + 8 + i] | b[at + 9 + i] << 8) / 32768.0f);
            }
            return true;
        }
        at += 8 + size;
    }
    return false;
}

// Turns the lines of whisper_print_timings, `  encode time =  690.75 ms / 1 runs`, into `phase NAME MS` lines.
static void timing_log(ggml_log_level, const char * text, void *) {
    const char * at = strstr(text, " time = ");
    char name[32];
    double ms;
    if (at && sscanf(text, "%*[^:]: %31s time = %lf ms", name, &ms) == 2) printf("phase\t%s\t%.1f\n", name, ms);
}

static void transcribe(whisper_context * ctx, const whisper_full_params & wp, const std::vector<float> & pcm) {
    if (whisper_full(ctx, wp, pcm.data(), (int) pcm.size()) != 0) {
        fprintf(stderr, "whisper_full failed\n");
        exit(1);
    }
}

int main(int argc, char ** argv) {
    if (argc < 7) {
        fprintf(stderr, "usage: %s MODEL.bin CLIP.wav LANGUAGE THREADS USE_GPU FLASH_ATTN [RUNS]\n", argv[0]);
        return 2;
    }
    const char * language = argv[3];
    const int threads = atoi(argv[4]);
    const int runs = argc > 7 ? atoi(argv[7]) : 5;
    std::vector<float> pcm;
    if (runs >= 0 && !read_wav(argv[2], pcm)) {
        fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }

    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = atoi(argv[5]) != 0;
    cp.flash_attn = atoi(argv[6]) != 0;
    double t = now_ms();
    whisper_context * ctx = whisper_init_from_file_with_params(argv[1], cp);
    if (!ctx) return 1;
    printf("load\t%.1f\n", now_ms() - t);
    if (runs < 0) return 0;  // RUNS -1: stop after the load

    whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.n_threads = threads;
    wp.language = language;
    wp.detect_language = false;
    wp.no_timestamps = true;
    wp.print_progress = wp.print_realtime = wp.print_timestamps = wp.print_special = false;
    wp.temperature_inc = 0.0f;  // -nf: no temperature fallback
    wp.greedy.best_of = 1;      // -bo 1

    t = now_ms();
    transcribe(ctx, wp, pcm);
    printf("warmup\t%.1f\n", now_ms() - t);
    whisper_reset_timings(ctx);
    for (int i = 0; i < runs; i++) {
        t = now_ms();
        transcribe(ctx, wp, pcm);
        printf("run\t%.1f\n", now_ms() - t);
    }

    // whisper.cpp's own totals for the timed runs (mel, sample, encode, decode, batchd, prompt), as `phase` lines.
    if (runs > 0) {
        whisper_log_set(timing_log, nullptr);
        whisper_print_timings(ctx);
        whisper_log_set(nullptr, nullptr);
    }

    std::string text, tokens;
    for (int s = 0; s < whisper_full_n_segments(ctx); s++) {
        text += whisper_full_get_segment_text(ctx, s);
        for (int k = 0; k < whisper_full_n_tokens(ctx, s); k++) {
            whisper_token id = whisper_full_get_token_id(ctx, s, k);
            if (id < whisper_token_eot(ctx)) tokens += (tokens.empty() ? "" : " ") + std::to_string(id);
        }
    }
    printf("text\t%s\ntokens\t%s\n", text.c_str(), tokens.c_str());
    whisper_free(ctx);
    return 0;
}
