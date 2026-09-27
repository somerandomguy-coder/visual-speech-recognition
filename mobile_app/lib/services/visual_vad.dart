import 'dart:math';

/// Point representation for 2D facial landmark coordinates.
class Point2D {
  final double x;
  final double y;

  const Point2D(this.x, this.y);

  double distanceTo(Point2D other) {
    final dx = x - other.x;
    final dy = y - other.y;
    return sqrt(dx * dx + dy * dy);
  }
}

/// Visual Voice Activity Detection (V-VAD) based on Lip Aspect Ratio (LAR) and motion dynamics.
class VisualVAD {
  final double larThreshold;
  final double motionThreshold;
  final double speechOnsetMs;
  final double silenceDurationMs;

  bool _isSpeaking = false;
  double? _speechStartTimeMs;
  double? _silenceStartTimeMs;
  bool _phraseComplete = false;

  VisualVAD({
    this.larThreshold = 0.15,
    this.motionThreshold = 8.0,
    this.speechOnsetMs = 120.0,
    this.silenceDurationMs = 400.0,
  });

  bool get isSpeaking => _isSpeaking;
  bool get isPhraseComplete => _phraseComplete;

  void reset() {
    _isSpeaking = false;
    _speechStartTimeMs = null;
    _silenceStartTimeMs = null;
    _phraseComplete = false;
  }

  /// Update VAD state given key mouth landmark points [topLip, bottomLip, leftCorner, rightCorner].
  bool update({
    required Point2D topLip,
    required Point2D bottomLip,
    required Point2D leftCorner,
    required Point2D rightCorner,
    required double timestampMs,
  }) {
    final verticalDist = topLip.distanceTo(bottomLip);
    final horizontalDist = leftCorner.distanceTo(rightCorner);
    final lar = horizontalDist > 0.0 ? verticalDist / horizontalDist : 0.0;

    final mouthOpen = lar >= larThreshold;

    if (mouthOpen) {
      _silenceStartTimeMs = null;
      if (_speechStartTimeMs == null) {
        _speechStartTimeMs = timestampMs;
      } else if (timestampMs - _speechStartTimeMs! >= speechOnsetMs) {
        _isSpeaking = true;
        _phraseComplete = false;
      }
    } else {
      _speechStartTimeMs = null;
      if (_isSpeaking) {
        if (_silenceStartTimeMs == null) {
          _silenceStartTimeMs = timestampMs;
        } else if (timestampMs - _silenceStartTimeMs! >= silenceDurationMs) {
          _isSpeaking = false;
          _phraseComplete = true;
          _silenceStartTimeMs = null;
        }
      }
    }

    return _isSpeaking;
  }

  /// Consume the phrase completion signal and reset it.
  bool takePhraseComplete() {
    final complete = _phraseComplete;
    _phraseComplete = false;
    return complete;
  }
}
