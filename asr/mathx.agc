## `exp`, `log`, `log10`, `sin` and `cos` from the C library (libm), for the
## log-mel frontend. Temporary: Agentica's `Math` has only `sqrt`; once it has
## these (agentica-lang/agentica#1925), this module goes. The functions follow C:
## `log(0)` is -Inf, a negative argument gives NaN, and nothing raises.
##
## The `.agc` is the one source of the signatures; after a change to the block,
## from `asr/`:
##
## ```sh
## agentica deps/native-gen/cli.agc mathx.agc --output native/mathx_native.h
## ```

extern module "mathx"
  def exp(x: Float) -> Float
  def log(x: Float) -> Float
  def log10(x: Float) -> Float
  def sin(x: Float) -> Float
  def cos(x: Float) -> Float
end
