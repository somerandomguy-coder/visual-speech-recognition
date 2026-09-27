import 'dart:math';

/// Resamples variable frame-rate camera feeds into a strictly monotonic 25.0 FPS timeline.
class TemporalResampler<T> {
  final double targetFps;
  final double targetIntervalMs;
  
  double? _firstTimestampMs;
  double _nextTargetTimeMs = 0.0;
  final List<_TimestampedFrame<T>> _buffer = [];

  TemporalResampler({this.targetFps = 25.0})
      : targetIntervalMs = 1000.0 / targetFps;

  void reset() {
    _buffer.clear();
    _firstTimestampMs = null;
    _nextTargetTimeMs = 0.0;
  }

  /// Ingest a frame with its capture timestamp in milliseconds and emit 25.0 FPS frames.
  List<T> addFrame(T frame, double timestampMs) {
    if (_firstTimestampMs == null) {
      _firstTimestampMs = timestampMs;
      _nextTargetTimeMs = 0.0;
    }

    final relativeTimeMs = timestampMs - _firstTimestampMs!;
    _buffer.add(_TimestampedFrame(frame: frame, timeMs: relativeTimeMs));

    final emitted = <T>[];

    // Emit frames while the buffer spans past the next target timestamp
    while (_buffer.length >= 2 && _buffer.last.timeMs >= _nextTargetTimeMs) {
      final bestFrame = _findNearestFrame(_nextTargetTimeMs);
      emitted.add(bestFrame);
      _nextTargetTimeMs += targetIntervalMs;

      // Drop obsolete frames older than the current target window
      while (_buffer.length > 2 && _buffer[1].timeMs < _nextTargetTimeMs - targetIntervalMs) {
        _buffer.removeAt(0);
      }
    }

    return emitted;
  }

  /// Flush remaining frames in the buffer at the conclusion of a video clip or phrase.
  List<T> flush() {
    final emitted = <T>[];
    if (_buffer.isEmpty) return emitted;

    final lastTimeMs = _buffer.last.timeMs;
    while (_nextTargetTimeMs <= lastTimeMs) {
      emitted.add(_findNearestFrame(_nextTargetTimeMs));
      _nextTargetTimeMs += targetIntervalMs;
    }

    _buffer.clear();
    return emitted;
  }

  T _findNearestFrame(double targetTimeMs) {
    var minDiff = double.infinity;
    T bestFrame = _buffer.first.frame;

    for (final item in _buffer) {
      final diff = (item.timeMs - targetTimeMs).abs();
      if (diff < minDiff) {
        minDiff = diff;
        bestFrame = item.frame;
      }
    }

    return bestFrame;
  }
}

class _TimestampedFrame<T> {
  final T frame;
  final double timeMs;

  const _TimestampedFrame({required this.frame, required this.timeMs});
}
