/* Reads little-endian floats from a byte range of a file, or from bytes Agentica read (`TENSOR_LOAD=agentica`), into a
   host buffer of `float`, for `ggml.c` and `torch.c` (`Tensor.from_file`, `tensor/DESIGN.md` §4). It is compiled as C
   and as C++, so it casts every `malloc`.

   The range is checked against the size of the file before anything is read, so a short file, a bad offset or a
   directory is an error and never a read past the end. An F32 range is read straight into `dest`; F16 and BF16
   are read in chunks and widened into it, so no other buffer of the size of the tensor exists. A quantized range (the
   blocks of ggml's Q8_0, Q4_K and Q6_K) is dequantized the same way, or read as the blocks it is (`ff_read_bytes`)
   for a tensor that keeps them. */
#ifndef TENSOR_FLOAT_FILE_H
#define TENSOR_FLOAT_FILE_H

#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__)
#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "float_file.h reads the floats of a file in place: it needs a little-endian host"
#endif
#else
#error "float_file.h cannot tell the byte order of this host"
#endif

#ifdef __cplusplus
static_assert(sizeof(off_t) == 8, "float_file.h needs a 64-bit off_t");
#else
_Static_assert(sizeof(off_t) == 8, "float_file.h needs a 64-bit off_t");
#endif

/* The codes of `Types.file_code` in `src/types.agc`. */
enum { FF_F32 = 0, FF_F16 = 1, FF_BF16 = 2, FF_Q8_0 = 3, FF_U8 = 4, FF_Q4_K = 5, FF_Q6_K = 6 };

/* The values and the bytes of one block of a quantized format: Q8_0 32 values in 34 bytes, Q4_K and Q6_K 256 in 144 and
   210. Zero for a format that is not quantized. */
static int64_t ff_block_values(int64_t format) {
  switch (format) {
    case FF_Q8_0: return 32;
    case FF_Q4_K:
    case FF_Q6_K: return 256;
    default: return 0;
  }
}

static int64_t ff_block_bytes(int64_t format) {
  switch (format) {
    case FF_Q8_0: return 34;
    case FF_Q4_K: return 144;
    case FF_Q6_K: return 210;
    default: return 0;
  }
}

/* The raw bytes read at a time for F16 and BF16. */
#define FF_CHUNK ((size_t)1 << 20)
#define FF_PATH_MAX 4096

static size_t ff_width(int64_t format) {
  switch (format) {
    case FF_F32: return 4;
    case FF_F16: return 2;
    case FF_BF16: return 2;
    default: return 0;
  }
}

/* The binary32 of a binary16, exactly: subnormals are normalised, infinities and NaNs keep their payload. */
static float ff_half(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  float out;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 113; /* 127 - 15 + 1: the exponent of the first subnormal */
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        exp--;
      }
      bits = sign | (exp << 23) | ((mant & 0x3FFu) << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  memcpy(&out, &bits, sizeof out);
  return out;
}

static float ff_bfloat(uint16_t b) {
  uint32_t bits = (uint32_t)b << 16;
  float out;

  memcpy(&out, &bits, sizeof out);
  return out;
}

/* `count` values of a 16-bit `format` (FF_F16 or FF_BF16) in `raw`, widened into `dest`. */
static void ff_widen(const uint8_t *raw, int64_t count, int64_t format, float *dest) {
  int64_t i;

  for (i = 0; i < count; i++) {
    uint16_t v = (uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8));

    dest[i] = format == FF_F16 ? ff_half(v) : ff_bfloat(v);
  }
}

/* The scale and the minimum of sub-block `j` of a Q4_K block: six bits each, packed in 12 bytes as ggml does. */
static void ff_scale_min_k4(int j, const uint8_t *q, uint8_t *scale, uint8_t *min) {
  if (j < 4) {
    *scale = q[j] & 63;
    *min = q[j + 4] & 63;
  } else {
    *scale = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
    *min = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
  }
}

/* One Q4_K block: d and dmin (binary16), 12 bytes of scales, 128 bytes of nibbles. Same operations as ggml. */
static void ff_dequant_q4_k(const uint8_t *block, float *y) {
  const uint8_t *q = block + 16;
  float d = ff_half((uint16_t)(block[0] | (block[1] << 8)));
  float min = ff_half((uint16_t)(block[2] | (block[3] << 8)));
  int j, l, is = 0;
  uint8_t sc, m;

  for (j = 0; j < 256; j += 64) {
    float d1, m1, d2, m2;

    ff_scale_min_k4(is + 0, block + 4, &sc, &m);
    d1 = d * sc;
    m1 = min * m;
    ff_scale_min_k4(is + 1, block + 4, &sc, &m);
    d2 = d * sc;
    m2 = min * m;
    for (l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
    for (l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
    q += 32;
    is += 2;
  }
}

/* One Q6_K block: 128 bytes of low nibbles, 64 of high bit pairs, 16 signed scales, d (binary16). */
static void ff_dequant_q6_k(const uint8_t *block, float *y) {
  const uint8_t *ql = block, *qh = block + 128;
  const int8_t *sc = (const int8_t *)(block + 192);
  float d = ff_half((uint16_t)(block[208] | (block[209] << 8)));
  int n, l;

  for (n = 0; n < 256; n += 128) {
    for (l = 0; l < 32; l++) {
      int is = l / 16;
      int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
      int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
      int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
      int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;

      y[l + 0] = d * sc[is + 0] * q1;
      y[l + 32] = d * sc[is + 2] * q2;
      y[l + 64] = d * sc[is + 4] * q3;
      y[l + 96] = d * sc[is + 6] * q4;
    }
    y += 128;
    ql += 64;
    qh += 32;
    sc += 8;
  }
}

/* `blocks` blocks of the quantized `format` in `raw`, dequantized into `dest` as ggml does. Q8_0 is the scale (binary16,
   widened) times the byte. */
static void ff_dequant(int64_t format, const uint8_t *raw, int64_t blocks, float *dest) {
  size_t bytes = (size_t)ff_block_bytes(format);
  int64_t b, values = ff_block_values(format);
  int j;

  for (b = 0; b < blocks; b++) {
    const uint8_t *block = raw + (size_t)b * bytes;
    float *y = dest + b * values;

    if (format == FF_Q4_K) {
      ff_dequant_q4_k(block, y);
    } else if (format == FF_Q6_K) {
      ff_dequant_q6_k(block, y);
    } else {
      float scale = ff_half((uint16_t)(block[0] | (block[1] << 8)));

      for (j = 0; j < 32; j++) y[j] = scale * (float)(int8_t)block[2 + j];
    }
  }
}

/* The bytes of `count` values of `format`, or -1 when `format` is unknown or `count` is not whole blocks. */
static int64_t ff_bytes(int64_t format, int64_t count) {
  int64_t values = ff_block_values(format);

  if (values != 0) return count % values == 0 ? count / values * ff_block_bytes(format) : -1;
  return ff_width(format) == 0 ? -1 : count * (int64_t)ff_width(format);
}

/* What an `open` failure means, in words that do not need `strerror` (not thread safe). */
static const char *ff_open_why(int err) {
  switch (err) {
    case ENOENT: return "no such file";
    case EACCES: return "permission denied";
    case EISDIR: return "is a directory";
    case ENOTDIR: return "a part of the path is not a directory";
    case ENAMETOOLONG: return "the name is too long";
    case ELOOP: return "too many links";
    case EMFILE:
    case ENFILE: return "too many open files";
    default: return "cannot be opened";
  }
}

/* Reads exactly `len` bytes at `at` of `fd`. */
static bool ff_pread(int fd, void *into, size_t len, int64_t at) {
  uint8_t *p = (uint8_t *)into;
  ssize_t got;

  while (len > 0) {
    got = pread(fd, p, len, (off_t)at);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) return false;
    p += got;
    at += got;
    len -= (size_t)got;
  }
  return true;
}

/* The file `path` (`path_len` bytes, no NUL) opened for the range of `count` values of `width` bytes at byte `offset`:
   the descriptor, or -1 with the reason in `why`. The range is checked against the size of the file here. */
static int ff_open_range(const char *path, size_t path_len, int64_t offset, int64_t count, size_t width, char *why,
                         size_t why_len) {
  char name[FF_PATH_MAX];
  struct stat info;
  int64_t bytes;
  int fd;

  if (offset < 0 || count < 0) {
    snprintf(why, why_len, "a range starts at 0 or later and holds no negative count, got offset %lld and %lld values",
             (long long)offset, (long long)count);
    return -1;
  }
  if (path_len == 0 || path_len >= sizeof name || memchr(path, 0, path_len) != NULL) {
    snprintf(why, why_len, "not a usable file path");
    return -1;
  }
  if (count > (INT64_MAX - offset) / (int64_t)width) {
    snprintf(why, why_len, "the range of %lld values at %lld is too large", (long long)count, (long long)offset);
    return -1;
  }
  memcpy(name, path, path_len);
  name[path_len] = '\0';

  /* O_NONBLOCK so that opening a FIFO does not wait for a writer: the `S_ISREG` check refuses it next. */
  fd = open(name, O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) {
    snprintf(why, why_len, "cannot open %s: %s", name, ff_open_why(errno));
    return -1;
  }
  if (fstat(fd, &info) != 0) {
    close(fd);
    snprintf(why, why_len, "cannot stat %s", name);
    return -1;
  }
  if (!S_ISREG(info.st_mode)) {
    close(fd);
    snprintf(why, why_len, "%s is not a regular file", name);
    return -1;
  }
  bytes = count * (int64_t)width;
  if (offset + bytes > (int64_t)info.st_size) {
    close(fd);
    snprintf(why, why_len, "%s has %lld bytes: the range of %lld bytes at %lld ends past them", name,
             (long long)info.st_size, (long long)bytes, (long long)offset);
    return -1;
  }
  return fd;
}

/* The `len` bytes at `offset` of the file `path`, as they are, into `dest`. False with the reason in `why`. */
static inline bool ff_read_bytes(const char *path, size_t path_len, int64_t offset, int64_t len, uint8_t *dest, char *why,
                          size_t why_len) {
  int fd = ff_open_range(path, path_len, offset, len, 1, why, why_len);

  if (fd < 0) return false;
  if (len > 0 && !ff_pread(fd, dest, (size_t)len, offset)) {
    close(fd);
    snprintf(why, why_len, "%.*s ended while it was read (it shrank?)", (int)path_len, path);
    return false;
  }
  close(fd);
  return true;
}

/* What a failed `write` or `close` means, in words that do not need `strerror` (not thread safe). */
static const char *ff_write_why(int err) {
  switch (err) {
    case ENOSPC: return "no space left on the device";
    case EDQUOT: return "the disk quota is used up";
    case EFBIG: return "the file would be too large";
    case EIO: return "an I/O error";
    case EROFS: return "the file system is read-only";
    default: return "the write failed";
  }
}

/* Appends `len` bytes to the regular file `path`, which must exist (`Tensor.append_file`). A failed write cuts the
   file back to the size it had; a failed `close` (where a network file system or a quota reports it) is a failure. */
static inline bool ff_append(const char *path, size_t path_len, const void *from, size_t len, char *why, size_t why_len) {
  const uint8_t *p = (const uint8_t *)from;
  char name[FF_PATH_MAX];
  struct stat info;
  ssize_t put;
  int fd, cut, err = 0;

  if (path_len == 0 || path_len >= sizeof name || memchr(path, 0, path_len) != NULL) {
    snprintf(why, why_len, "not a usable file path");
    return false;
  }
  memcpy(name, path, path_len);
  name[path_len] = '\0';
  fd = open(name, O_WRONLY | O_APPEND | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) {
    snprintf(why, why_len, "cannot open %s: %s", name, ff_open_why(errno));
    return false;
  }
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
    close(fd);
    snprintf(why, why_len, "%s is not a regular file", name);
    return false;
  }
  while (len > 0 && err == 0) {
    put = write(fd, p, len);
    if (put < 0 && errno == EINTR) continue;
    if (put < 0) err = errno;
    else if (put == 0) err = EIO;
    else {
      p += put;
      len -= (size_t)put;
    }
  }
  if (err != 0) {
    /* The write error is the one reported, whether or not the cut works. */
    cut = ftruncate(fd, info.st_size);
    (void)cut;
  }
  if (close(fd) != 0 && err == 0) err = errno;
  if (err != 0) {
    snprintf(why, why_len, "cannot write %s: %s", name, ff_write_why(err));
    return false;
  }
  return true;
}

/* The `count` values of a quantized range (whole blocks) dequantized into `dest`, a chunk of blocks at a time. */
static bool ff_read_blocks(const char *path, size_t path_len, int64_t offset, int64_t count, int64_t format, float *dest,
                           char *why, size_t why_len) {
  int64_t values = ff_block_values(format), bytes = ff_block_bytes(format);
  int64_t blocks = count / values, per_chunk = (int64_t)FF_CHUNK / bytes, done = 0, n;
  uint8_t *raw;
  int fd;

  if (count % values != 0) {
    snprintf(why, why_len, "%lld values are not whole blocks of %lld", (long long)count, (long long)values);
    return false;
  }
  fd = ff_open_range(path, path_len, offset, blocks, (size_t)bytes, why, why_len);
  if (fd < 0) return false;
  raw = (uint8_t *)calloc((size_t)per_chunk, (size_t)bytes);
  if (raw == NULL) {
    close(fd);
    snprintf(why, why_len, "out of memory");
    return false;
  }
  while (done < blocks) {
    n = blocks - done < per_chunk ? blocks - done : per_chunk;
    if (!ff_pread(fd, raw, (size_t)(n * bytes), offset + done * bytes)) {
      free(raw);
      close(fd);
      snprintf(why, why_len, "%.*s ended while it was read (it shrank?)", (int)path_len, path);
      return false;
    }
    ff_dequant(format, raw, n, dest + done * values);
    done += n;
  }
  free(raw);
  close(fd);
  return true;
}

/* The `count` values of `format` that start at byte `offset` of the file `path` (`path_len` bytes, no NUL), as
   floats in `dest` (room for `count`; it may be NULL when `count` is 0). False with the reason in `why`. */
static bool ff_read(const char *path, size_t path_len, int64_t offset, int64_t count, int64_t format, float *dest,
                    char *why, size_t why_len) {
  size_t width = ff_width(format), step, n;
  int64_t done = 0;
  uint8_t *raw;
  int fd;

  if (ff_block_values(format) != 0) return ff_read_blocks(path, path_len, offset, count, format, dest, why, why_len);
  if (width == 0) {
    snprintf(why, why_len, "unknown float format code %lld", (long long)format);
    return false;
  }
  fd = ff_open_range(path, path_len, offset, count, width, why, why_len);
  if (fd < 0) return false;
  if (count == 0) {
    close(fd);
    return true;
  }

  if (format == FF_F32) {
    if (!ff_pread(fd, dest, (size_t)count * 4, offset)) {
      close(fd);
      snprintf(why, why_len, "%.*s ended while it was read (it shrank?)", (int)path_len, path);
      return false;
    }
    close(fd);
    return true;
  }

  /* the chunk, or all of a smaller range */
  n = count < (int64_t)(FF_CHUNK / 2) ? (size_t)count : FF_CHUNK / 2;
  raw = (uint8_t *)calloc(n, 2); /* zeroed: the analyzers cannot see that ff_pread fills it */
  if (raw == NULL) {
    close(fd);
    snprintf(why, why_len, "out of memory");
    return false;
  }
  while (done < count) {
    n = FF_CHUNK / 2;
    if ((int64_t)n > count - done) n = (size_t)(count - done);
    step = n * 2;
    if (!ff_pread(fd, raw, step, offset + done * 2)) {
      free(raw);
      close(fd);
      snprintf(why, why_len, "%.*s ended while it was read (it shrank?)", (int)path_len, path);
      return false;
    }
    ff_widen(raw, (int64_t)n, format, dest + done);
    done += (int64_t)n;
  }
  free(raw);
  close(fd);
  return true;
}

/* The `count` values of `format` held by the `len` bytes of `data` (which Agentica read from a file), as floats in
   `dest` (room for `count`; it may be NULL when `count` is 0). `len` must be exactly the range. False with the reason
   in `why`. */
static bool ff_from_bytes(const uint8_t *data, size_t len, int64_t count, int64_t format, float *dest, char *why,
                          size_t why_len) {
  size_t width = ff_width(format);

  if (format == FF_U8) {
    int64_t i;

    if (count < 0 || (uint64_t)count != len) {
      snprintf(why, why_len, "%zu bytes are not %lld U8 values", len, (long long)count);
      return false;
    }
    for (i = 0; i < count; i++) dest[i] = (float)data[i];
    return true;
  }
  if (ff_block_values(format) != 0) {
    if (count < 0 || count % ff_block_values(format) != 0 || len != (size_t)ff_bytes(format, count)) {
      snprintf(why, why_len, "%zu bytes are not %lld values of whole blocks of %lld bytes", len, (long long)count,
               (long long)ff_block_bytes(format));
      return false;
    }
    ff_dequant(format, data, count / ff_block_values(format), dest);
    return true;
  }
  if (width == 0) {
    snprintf(why, why_len, "unknown float format code %lld", (long long)format);
    return false;
  }
  if (count < 0 || (uint64_t)count > SIZE_MAX / width || len != (size_t)count * width) {
    snprintf(why, why_len, "%zu bytes are not %lld values of %zu bytes", len, (long long)count, width);
    return false;
  }
  if (count == 0) return true;
  if (format == FF_F32) {
    memcpy(dest, data, len);
  } else {
    ff_widen(data, count, format, dest);
  }
  return true;
}

/* The values read in `FF_VALUES` at a time, which `ff_each` hands to a sink: `first` is the index of the first value
   of `values`, `n` how many. The sink answers false, with the reason in `why`, to stop. */
#define FF_VALUES (FF_CHUNK / 2)
typedef bool (*ff_sink)(void *ctx, const float *values, int64_t first, int64_t n, char *why, size_t why_len);

/* The values `begin` to `end` (a part of the `count` that start at byte `offset`) of the open file `fd`, handed to `sink`
   a chunk at a time as floats. `path` is only for the message of a short file. The run number `mine` and `lowest`, the
   lowest number of a run that failed (shared by the threads, NULL for a lone run), let a run stop between chunks once
   an earlier run has failed: it can no longer be the first failure in file order. */
static inline bool ff_each_span(int fd, const char *path, size_t path_len, int64_t offset, int64_t begin, int64_t end,
                                int64_t format, ff_sink sink, void *ctx, int mine, int *lowest, char *why,
                                size_t why_len) {
  size_t width = ff_width(format);
  int64_t done = begin, n;
  uint8_t *raw;
  float *values;
  bool ok = true;

  if (width == 0) {
    snprintf(why, why_len, "unknown float format code %lld", (long long)format);
    return false;
  }
  n = end - begin < (int64_t)FF_VALUES ? end - begin : (int64_t)FF_VALUES;
  raw = (uint8_t *)calloc((size_t)n + 1, width);
  values = (float *)calloc((size_t)n + 1, sizeof(float));
  if (raw == NULL || values == NULL) {
    snprintf(why, why_len, "out of memory");
    ok = false;
  }
  while (ok && done < end && (lowest == NULL || __atomic_load_n(lowest, __ATOMIC_RELAXED) > mine)) {
    n = end - done < (int64_t)FF_VALUES ? end - done : (int64_t)FF_VALUES;
    if (!ff_pread(fd, raw, (size_t)n * width, offset + done * (int64_t)width)) {
      snprintf(why, why_len, "%.*s ended while it was read (it shrank?)", (int)path_len, path);
      ok = false;
    } else {
      if (format == FF_F32) memcpy(values, raw, (size_t)n * 4);
      else ff_widen(raw, n, format, values);
      ok = sink(ctx, values, done, n, why, why_len);
      done += n;
    }
  }
  free(raw);
  free(values);
  if (!ok && lowest != NULL) {
    int seen = __atomic_load_n(lowest, __ATOMIC_RELAXED);

    while (mine < seen) {
      if (__atomic_compare_exchange_n(lowest, &seen, mine, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) break;
    }
  }
  return ok;
}

/* The `count` values of `format` at byte `offset` of the file `path`, one open of the file for the whole range, handed
   to `sink` a chunk at a time as floats: no buffer of the size of the tensor exists. With `threads` above 1 the range is
   cut into runs of whole chunks, one per thread, and `sink` is called from several threads at once: it must only write
   what its `first` and `n` name. The first failure in file order is the one reported. */
struct ff_job {
  int fd;
  const char *path;
  size_t path_len;
  int64_t offset, begin, end, format;
  ff_sink sink;
  void *ctx;
  int mine;
  int *lowest;
  char *why;
  size_t why_len;
  bool ok;
};

static void *ff_job_run(void *arg) {
  struct ff_job *job = (struct ff_job *)arg;

  job->ok = ff_each_span(job->fd, job->path, job->path_len, job->offset, job->begin, job->end, job->format, job->sink,
                         job->ctx, job->mine, job->lowest, job->why, job->why_len);
  return NULL;
}

#define FF_MAX_THREADS 16

static inline bool ff_each(const char *path, size_t path_len, int64_t offset, int64_t count, int64_t format,
                           ff_sink sink, void *ctx, int threads, char *why, size_t why_len) {
  size_t width = ff_width(format);
  struct ff_job jobs[FF_MAX_THREADS];
  pthread_t ids[FF_MAX_THREADS] = {0};
  bool started[FF_MAX_THREADS];
  char *whys[FF_MAX_THREADS];
  int64_t chunks, k;
  bool ok = true;
  int fd, t, used, lowest;

  if (width == 0) {
    snprintf(why, why_len, "unknown float format code %lld", (long long)format);
    return false;
  }
  fd = ff_open_range(path, path_len, offset, count, width, why, why_len);
  if (fd < 0) return false;
  chunks = (count + (int64_t)FF_VALUES - 1) / (int64_t)FF_VALUES;
  used = threads < 1 ? 1 : (threads > FF_MAX_THREADS ? FF_MAX_THREADS : threads);
  if ((int64_t)used > chunks) used = chunks < 1 ? 1 : (int)chunks;
  if (used == 1) {
    ok = ff_each_span(fd, path, path_len, offset, 0, count, format, sink, ctx, 0, NULL, why, why_len);
    close(fd);
    return ok;
  }

  lowest = used;
  for (t = 0; t < used; t++) {
    jobs[t].fd = fd;
    jobs[t].path = path;
    jobs[t].path_len = path_len;
    jobs[t].offset = offset;
    jobs[t].format = format;
    jobs[t].sink = sink;
    jobs[t].ctx = ctx;
    jobs[t].why_len = why_len;
    jobs[t].ok = false;
    jobs[t].mine = t;
    jobs[t].lowest = &lowest;
    jobs[t].begin = chunks * t / used * (int64_t)FF_VALUES;
    k = chunks * (t + 1) / used * (int64_t)FF_VALUES;
    jobs[t].end = k < count ? k : count;
    whys[t] = (char *)calloc(why_len, 1);
    jobs[t].why = whys[t];
    started[t] = false;
    if (whys[t] == NULL) {
      snprintf(why, why_len, "out of memory");
      ok = false;
    }
  }
  for (t = 0; ok && t < used; t++) {
    started[t] = pthread_create(&ids[t], NULL, ff_job_run, &jobs[t]) == 0;
    if (!started[t]) ff_job_run(&jobs[t]);
  }
  for (t = 0; t < used; t++) {
    if (started[t]) pthread_join(ids[t], NULL);
  }
  for (t = 0; ok && t < used; t++) {
    if (!jobs[t].ok) {
      snprintf(why, why_len, "%s", whys[t]);
      ok = false;
    }
  }
  for (t = 0; t < used; t++) free(whys[t]);
  close(fd);
  return ok;
}

/* The same over the `len` bytes `data` that Agentica read, which must be exactly the range. */
static inline bool ff_each_bytes(const uint8_t *data, size_t len, int64_t count, int64_t format, ff_sink sink, void *ctx,
                                 char *why, size_t why_len) {
  size_t width = ff_width(format);
  int64_t done = 0, n;
  float *values;
  bool ok = true;

  if (width == 0) {
    snprintf(why, why_len, "unknown float format code %lld", (long long)format);
    return false;
  }
  if (count < 0 || (uint64_t)count > SIZE_MAX / width || len != (size_t)count * width) {
    snprintf(why, why_len, "%zu bytes are not %lld values of %zu bytes", len, (long long)count, width);
    return false;
  }
  n = count < (int64_t)FF_VALUES ? count : (int64_t)FF_VALUES;
  values = (float *)calloc((size_t)n + 1, sizeof(float));
  if (values == NULL) {
    snprintf(why, why_len, "out of memory");
    return false;
  }
  while (ok && done < count) {
    n = count - done < (int64_t)FF_VALUES ? count - done : (int64_t)FF_VALUES;
    if (format == FF_F32) memcpy(values, data + (size_t)done * 4, (size_t)n * 4);
    else ff_widen(data + (size_t)done * 2, n, format, values);
    ok = sink(ctx, values, done, n, why, why_len);
    done += n;
  }
  free(values);
  return ok;
}

#endif
