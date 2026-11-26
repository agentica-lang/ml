## Long-audio transcription for whisper: audio longer than the five-minute
## bound of one `transcribe_with` call is transcribed piecewise, in bounded
## windows with overlap, and the windows' segments are merged back onto one
## timeline.
##
## `StreamOptions` is how one `transcribe_stream` call walks the audio: 30 s
## windows with 5 s of overlap by default (`default_options`), each window a
## separate `transcribe_with` call (which is where the long work runs: each
## decode stays in the compute pool, so no scheduler is blocked), plus the
## energy gate and the engine options. `check_options` validates the walk
## without touching audio or a model.
##
## The gate is pure Agentica RMS energy per frame: `frame_rms` over
## `frame_ms` (30 ms, 480 samples by default), `is_speech` when that RMS
## reaches `threshold` (0.02, about -34 dBFS: a quiet room sits below it,
## conversational speech sits above it). `has_speech` keeps a window when any
## frame is speech, so silent windows cost no engine call; `threshold: 0.0`
## keeps every window and turns the gate off. Very quiet speech below the
## threshold is skipped with the silence: tune the threshold down (or off)
## rather than up when words go missing.
##
## `windows` cuts the PCM into the bounded stretches (the last one shorter),
## `offset_segments` moves one window's segments onto the stream timeline, and
## `merge_segments` sorts the offset segments and drops the overlap doubles:
## an empty text, or a segment whose stripped text matches (case aside) an
## already kept segment it overlaps in time. Distinct hypotheses in the
## overlap are all kept: the merge never drops words, it only drops copies.
##
## `transcribe_stream` is synchronous batch-over-a-stream: it reads the whole
## `Source` (synthetic PCM or one microphone capture), transcribes each
## speech window, offsets and merges. There is no incremental `Channel`
## variant on purpose: one would need a producer task outliving the call,
## which breaks this package's rule that no call leaves a task or a process
## running. Latency is the sum of the window decodes, and every returned
## segment is final (each window is decoded whole): there are no partial
## hypotheses. `whisper.cpp`'s own silero VAD (`whisper_vad_*`) stays out for
## the same reason: it needs a separate VAD model download plus C wiring for
## a gate this pure function already covers, so the pure gate is the only one
## until a measurement says otherwise.
##
## `transcribe` and `transcribe_with` are untouched: this file only adds the
## walk over them.

import Whisper from "whisper"
import Mic from "mic"

## The energy gate: each `frame_ms` of audio is speech when its RMS reaches
## `threshold` (0.0..1.0, where 0.0 keeps every window).
struct VadOptions
  frame_ms: Int = 30
  threshold: Float = 0.02
end

## How one `transcribe_stream` call walks the audio: windows of `window_s`
## seconds starting every `window_s - overlap_s` seconds, gated by `vad` and
## decoded as `options` asks. `window_s` holds 1..300 s (one engine call per
## window, five minutes at most) and `overlap_s` holds 0..window_s-1 s.
struct StreamOptions
  window_s: Int = 30
  overlap_s: Int = 5
  vad: VadOptions
  options: Whisper.Options
end

## One bounded stretch of the stream: `samples` starting at `start_sample`
## (in 16 kHz samples from the first sample of the stream).
struct Window
  start_sample: Int
  samples: List[Float32]
end

## The default walk: 30 s windows with 5 s of overlap, the 30 ms / 0.02 energy
## gate and the engine's own default options.
def default_options() -> StreamOptions
  StreamOptions(window_s: 30, overlap_s: 5, vad: VadOptions(), options: Whisper.Options())
end

## Checks the walk without touching audio or a model: error on a window past
## five minutes, an overlap outside its window, a frame outside 1..1000 ms or
## a threshold outside 0.0..1.0.
def check_options(options: StreamOptions) -> Result[Void]
  if options.window_s < 1 or options.window_s > 300
    return error("stream: window_s holds 1..300 s (five minutes at most)")
  end

  if options.overlap_s < 0 or options.overlap_s >= options.window_s
    return error("stream: overlap_s holds 0..window_s-1 s")
  end

  return error("stream: frame_ms holds 1..1000 ms") if options.vad.frame_ms < 1 or options.vad.frame_ms > 1000
  return error("stream: threshold holds 0.0..1.0") if options.vad.threshold < 0.0 or options.vad.threshold > 1.0
  ok()
end

## The RMS energy of one frame: 0.0 for no samples, otherwise the square root
## of the mean square. Never an error, never a halt.
def frame_rms(frame: List[Float32]) -> Float
  return 0.0 if frame.empty?
  sum = 0.0
  for s in frame
    x = s.to_f
    sum += x * x
  end

  Math.sqrt(sum / frame.size.to_f)
end

## Whether one frame is speech: its RMS reaches the gate's threshold.
def is_speech(frame: List[Float32], vad: VadOptions) -> Bool = frame_rms(frame) >= vad.threshold

## Whether any `frame_ms` frame of `pcm` is speech (a trailing partial frame
## counts whole): false for no samples.
def has_speech(pcm: List[Float32], vad: VadOptions) -> Bool
  frame_samples = (vad.frame_ms * Mic.RATE_HZ) / 1000
  start = 0
  while start < pcm.size
    return true if is_speech(pcm.drop(start).take(frame_samples), vad)
    start += frame_samples
  end

  false
end

## Cuts `pcm` into windows of `window_samples` starting every
## `window_samples - overlap_samples` samples; the last window is shorter
## when the audio does not fill it. Error on an empty input or sizes outside
## 1..4800000 samples with the overlap inside its window.
def windows(pcm: List[Float32], window_samples: Int, overlap_samples: Int) -> Result[List[Window]]
  return error("stream: a window holds 1..4800000 samples") if window_samples < 1 or window_samples > Mic.MAX_SAMPLES
  if overlap_samples < 0 or overlap_samples >= window_samples
    return error("stream: an overlap holds 0..window_samples-1 samples")
  end

  return error("stream: windows need at least one sample at 16000 Hz") if pcm.empty?
  step = window_samples - overlap_samples
  out: List[Window] = []
  start = 0
  while start < pcm.size
    length = window_samples
    if start + length > pcm.size
      length = pcm.size - start
    end

    out = out.push(Window(start_sample: start, samples: pcm.drop(start).take(length)))
    start += step
  end

  ok(out)
end

## Moves one window's segments onto the stream timeline: each bound plus
## `base_ms` (the window's start in milliseconds).
def offset_segments(segments: List[Whisper.Segment], base_ms: Int) -> List[Whisper.Segment]
  segments.map do |segment|
    Whisper.Segment(start_ms: segment.start_ms + base_ms, end_ms: segment.end_ms + base_ms, text: segment.text)
  end
end

## Merges offset windows onto one timeline: sorted by start, then each
## segment kept unless its text is empty or an already kept segment overlaps
## it in time with the same stripped text (case aside). Pure: no audio, no
## model, safe to test offline.
def merge_segments(segments: List[Whisper.Segment]) -> List[Whisper.Segment]
  ordered = segments.sort_by do |segment| segment.start_ms end
  kept: List[Whisper.Segment] = []
  for segment in ordered
    if keep_segment(kept, segment)
      kept = kept.push(segment)
    end
  end

  kept
end

## Transcribes a whole `source` piecewise: reads it (synthetic PCM or one
## microphone capture), checks the walk, decodes each speech window as
## `options.options` asks, offsets each window onto the stream timeline and
## merges. Silent windows cost no engine call; an input with no speech at all
## is no segments. Long work stays in `transcribe_with` (the compute pool);
## this walk only slices, offsets and merges.
def transcribe_stream(
  model: Whisper.Model,
  source: Mic.Source,
  options: StreamOptions,
) -> Result[List[Whisper.Segment]]
  try check_options(options)
  pcm = try Mic.read(source)
  return error("stream: transcribe_stream needs at least one sample at 16000 Hz") if pcm.empty?
  found = try transcribe_windows(model, pcm, options)
  ok(merge_segments(found))
end

private

# Whether two segments share any millisecond.
def overlaps(first: Whisper.Segment, second: Whisper.Segment) -> Bool
  first.start_ms < second.end_ms and second.start_ms < first.end_ms
end

# Whether `segment` survives the merge against the segments already kept.
def keep_segment(kept: List[Whisper.Segment], segment: Whisper.Segment) -> Bool
  text = segment.text.strip
  return false if text.empty?
  lowered = text.downcase
  for done in kept
    return false if done.text.strip.downcase == lowered and overlaps(done, segment)
  end

  true
end

# Decodes each speech window of `pcm` and offsets it onto the stream
# timeline; silent windows are skipped without an engine call.
def transcribe_windows(
  model: Whisper.Model,
  pcm: List[Float32],
  options: StreamOptions,
) -> Result[List[Whisper.Segment]]
  wins = try windows(pcm, options.window_s * Mic.RATE_HZ, options.overlap_s * Mic.RATE_HZ)
  found: List[Whisper.Segment] = []
  for win in wins
    if has_speech(win.samples, options.vad)
      segments = try Whisper.transcribe_with(model, Whisper.widen(win.samples), options.options)
      found = found + offset_segments(segments, (win.start_sample * 1000) / Mic.RATE_HZ)
    end
  end

  ok(found)
end
