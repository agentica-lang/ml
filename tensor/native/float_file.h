/* Reads little-endian floats from a byte range of a file into a host buffer of `float`, for `ggml.c` and `torch.c`
   (`Tensor.from_file`, `tensor/DESIGN.md` §4). It is compiled as C and as C++, so it casts every `malloc`.

   The range is checked against the size of the file before anything is read, so a short file, a bad offset or a
   directory is an error and never a read past the end. An F32 range is read straight into `dest`; F16 and BF16
   are read in chunks and widened into it, so no other buffer of the size of the tensor exists. */
#ifndef TENSOR_FLOAT_FILE_H
#define TENSOR_FLOAT_FILE_H

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__BYTE_ORDER__) && defined(__ORDER_LITTLE_ENDIAN__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "float_file.h reads the floats of a file in place: it needs a little-endian host"
#endif

/* The codes of `Types.file_code` in `src/types.agc`. */
enum { FF_F32 = 0, FF_F16 = 1, FF_BF16 = 2 };

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

/* The `count` values of `format` that start at byte `offset` of the file `path` (`path_len` bytes, no NUL), as
   floats in `dest` (room for `count`; it may be NULL when `count` is 0). False with the reason in `why`. */
static bool ff_read(const char *path, size_t path_len, int64_t offset, int64_t count, int64_t format, float *dest,
                    char *why, size_t why_len) {
  char name[FF_PATH_MAX];
  struct stat info;
  size_t width = ff_width(format), step, i, n;
  int64_t done = 0;
  uint8_t *raw;
  int fd;

  if (width == 0) {
    snprintf(why, why_len, "unknown float format code %lld", (long long)format);
    return false;
  }
  if (offset < 0 || count < 0) {
    snprintf(why, why_len, "a range starts at 0 or later and holds no negative count, got offset %lld and %lld values",
             (long long)offset, (long long)count);
    return false;
  }
  if (path_len == 0 || path_len >= sizeof name || memchr(path, 0, path_len) != NULL) {
    snprintf(why, why_len, "not a usable file path");
    return false;
  }
  if (count > (INT64_MAX - offset) / (int64_t)width) {
    snprintf(why, why_len, "the range of %lld values at %lld is too large", (long long)count, (long long)offset);
    return false;
  }
  memcpy(name, path, path_len);
  name[path_len] = '\0';

  fd = open(name, O_RDONLY);
  if (fd < 0) {
    snprintf(why, why_len, "cannot open %s: %s", name, strerror(errno));
    return false;
  }
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
    close(fd);
    snprintf(why, why_len, "%s is not a regular file", name);
    return false;
  }
  if (offset + count * (int64_t)width > (int64_t)info.st_size) {
    close(fd);
    snprintf(why, why_len, "%s has %lld bytes: the range of %lld bytes at %lld ends past them", name,
             (long long)info.st_size, (long long)(count * (int64_t)width), (long long)offset);
    return false;
  }
  if (count == 0) {
    close(fd);
    return true;
  }

  if (format == FF_F32) {
    if (!ff_pread(fd, dest, (size_t)count * 4, offset)) {
      close(fd);
      snprintf(why, why_len, "%s ended while it was read (it shrank?)", name);
      return false;
    }
    close(fd);
    return true;
  }

  raw = (uint8_t *)malloc(FF_CHUNK);
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
      snprintf(why, why_len, "%s ended while it was read (it shrank?)", name);
      return false;
    }
    for (i = 0; i < n; i++) {
      uint16_t v = (uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8));

      dest[done + (int64_t)i] = format == FF_F16 ? ff_half(v) : ff_bfloat(v);
    }
    done += (int64_t)n;
  }
  free(raw);
  close(fd);
  return true;
}

#endif
