## The streaming layer with no model: the default walk, its validation, the
## energy gate, the window boundaries, the timestamp offsetting and the
## overlap merge. Real decodes are in `live/stream_live_test.agc`.

import Stream from "../stream"
import Whisper from "../whisper"

private

def test_default_options_are_the_documented_walk() -> Result[Void]
  options = Stream.default_options()
  try Test.eq(options.window_s, 30)
  try Test.eq(options.overlap_s, 5)
  try Test.eq(options.vad.frame_ms, 30)
  try Test.eq(options.vad.threshold, 0.02)
  try Test.eq(options.options.language, "")
  Test.eq(options.options.translate, false)
end

def test_check_options_accepts_the_default_walk() -> Result[Void]
  Test.ok?(Stream.check_options(Stream.default_options()))
end

def test_check_options_rejects_a_window_outside_five_minutes() -> Result[Void]
  case Stream.check_options(Stream.default_options().with(window_s: 0))
    ok(_) -> Test.assert(false, "check_options took a zero window")
    error(e) -> Test.eq(e.message, "stream: window_s holds 1..300 s (five minutes at most)")
  end
end

def test_check_options_rejects_an_overlap_outside_its_window() -> Result[Void]
  case Stream.check_options(Stream.default_options().with(overlap_s: 30))
    ok(_) -> Test.assert(false, "check_options took an overlap of a whole window")
    error(e) -> Test.eq(e.message, "stream: overlap_s holds 0..window_s-1 s")
  end
end

def test_check_options_rejects_a_frame_outside_a_second() -> Result[Void]
  bad_frame = Stream.VadOptions(frame_ms: 0, threshold: 0.02)
  case Stream.check_options(Stream.default_options().with(vad: bad_frame))
    ok(_) -> Test.assert(false, "check_options took a zero frame")
    error(e) -> Test.eq(e.message, "stream: frame_ms holds 1..1000 ms")
  end
end

def test_check_options_rejects_a_threshold_outside_silence_and_full_scale() -> Result[Void]
  bad_threshold = Stream.VadOptions(frame_ms: 30, threshold: 2.0)
  case Stream.check_options(Stream.default_options().with(vad: bad_threshold))
    ok(_) -> Test.assert(false, "check_options took a threshold past 1.0")
    error(e) -> Test.eq(e.message, "stream: threshold holds 0.0..1.0")
  end
end

def test_frame_rms_measures_energy() -> Result[Void]
  try Test.eq(Stream.frame_rms([]), 0.0)
  try Test.eq(Stream.frame_rms([0.0, 0.0]), 0.0)
  try Test.eq(Stream.frame_rms([1.0, -1.0]), 1.0)
  Test.eq(Stream.frame_rms([0.5, 0.5]), 0.5)
end

def test_is_speech_compares_rms_against_the_threshold() -> Result[Void]
  strict = Stream.VadOptions(frame_ms: 30, threshold: 1.0)
  try Test.eq(Stream.is_speech([1.0, 1.0], strict), true)
  try Test.eq(Stream.is_speech([0.5, 0.5], strict), false)
  open = Stream.VadOptions(frame_ms: 30, threshold: 0.0)
  Test.eq(Stream.is_speech([0.0, 0.0], open), true)
end

def test_has_speech_needs_one_hot_frame() -> Result[Void]
  vad = Stream.VadOptions()
  frame: List[Float32] = []
  for _i in 0...480
    frame = frame.push(0.0)
  end

  try Test.eq(Stream.has_speech(frame, vad), false)
  try Test.eq(Stream.has_speech(frame.push(1.0), vad), true)
  try Test.eq(Stream.has_speech([1.0], vad), true)
  Test.eq(Stream.has_speech([], vad), false)
end

def test_windows_cover_the_pcm_with_overlap() -> Result[Void]
  pcm: List[Float32] = []
  for i in 0...100
    pcm = pcm.push(i.to_float32)
  end

  wins = try Stream.windows(pcm, 30, 10)
  try Test.eq(wins.size, 5)
  starts = wins.map do |win| win.start_sample end
  sizes = wins.map do |win| win.samples.size end
  try Test.eq(starts, [0, 20, 40, 60, 80])
  try Test.eq(sizes, [30, 30, 30, 30, 20])
  Test.eq(wins[0].samples, pcm.take(30))
end

def test_windows_keep_a_short_tail_and_a_short_stream() -> Result[Void]
  pcm: List[Float32] = []
  for i in 0...10
    pcm = pcm.push(i.to_float32)
  end

  short = try Stream.windows(pcm, 30, 10)
  try Test.eq(short.size, 1)
  try Test.eq(short[0].start_sample, 0)
  Test.eq(short[0].samples, pcm)
end

def test_windows_reject_bad_sizes_and_no_audio() -> Result[Void]
  try Test.error?(Stream.windows([1.0], 0, 0))
  try Test.error?(Stream.windows([1.0], 30, 30))
  try Test.error?(Stream.windows([1.0], 30, 31))
  Test.error?(Stream.windows([], 30, 10))
end

def test_offset_segments_shifts_by_the_window_start() -> Result[Void]
  segments = [Whisper.Segment(start_ms: 100, end_ms: 200, text: "hola")]
  shifted = Stream.offset_segments(segments, 5_000)
  try Test.eq(shifted.size, 1)
  try Test.eq(shifted[0].start_ms, 5_100)
  try Test.eq(shifted[0].end_ms, 5_200)
  try Test.eq(shifted[0].text, "hola")
  Test.eq(Stream.offset_segments([], 5_000), [])
end

def test_merge_segments_dedupes_same_text_overlaps() -> Result[Void]
  first = Whisper.Segment(start_ms: 0, end_ms: 1_000, text: "hola")
  echo = Whisper.Segment(start_ms: 200, end_ms: 900, text: "  HOLA  ")
  kept = Stream.merge_segments([echo, first])
  try Test.eq(kept.size, 1)
  Test.eq(kept[0].start_ms, 0)
end

def test_merge_segments_keeps_distinct_and_disjoint_text() -> Result[Void]
  a = Whisper.Segment(start_ms: 0, end_ms: 1_000, text: "hola")
  b = Whisper.Segment(start_ms: 500, end_ms: 1_500, text: "mundo")
  c = Whisper.Segment(start_ms: 5_000, end_ms: 6_000, text: "hola")
  kept = Stream.merge_segments([c, b, a])
  try Test.eq(kept.size, 3)
  starts = kept.map do |segment| segment.start_ms end
  Test.eq(starts, [0, 500, 5_000])
end

def test_merge_segments_drops_empty_text() -> Result[Void]
  segments = [Whisper.Segment(start_ms: 0, end_ms: 500, text: "   ")]
  Test.eq(Stream.merge_segments(segments), [])
end

def test_merge_segments_dedupes_exact_duplicates() -> Result[Void]
  segment = Whisper.Segment(start_ms: 1_000, end_ms: 2_000, text: "hola")
  kept = Stream.merge_segments([segment, segment])
  try Test.eq(kept.size, 1)
  Test.eq(kept[0], segment)
end

def test_window_offset_and_merge_cover_a_stream_end_to_end() -> Result[Void]
  wins = try Stream.windows(List.repeat(0.0, 1_000), 300, 100)
  try Test.eq(wins[2].start_sample, 400)
  first = Stream.offset_segments(
    [Whisper.Segment(start_ms: 0, end_ms: 500, text: "hola")],
    (wins[0].start_sample * 1000) / 16_000,
  )

  third = Stream.offset_segments(
    [
      Whisper.Segment(start_ms: 0, end_ms: 500, text: "hola"),
      Whisper.Segment(start_ms: 100, end_ms: 600, text: "mundo"),
    ],
    (wins[2].start_sample * 1000) / 16_000,
  )

  kept = Stream.merge_segments(first + third)
  try Test.eq(kept.size, 2)
  try Test.eq(kept[0].start_ms, 0)
  try Test.eq(kept[1].start_ms, 125)
  Test.eq(kept[1].text, "mundo")
end
