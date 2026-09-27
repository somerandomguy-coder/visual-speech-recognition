/// Subtitle event emitted by the visual speech recognition pipeline.
class SubtitleEvent {
  /// Tier 1: Instant visual phoneme tokens (<100ms) directly from ONNX CTC head.
  final String rawVisemes;

  /// Tier 2: Refined, punctuation-corrected English sentence from offline SLM.
  final String refinedText;

  /// Timestamp in milliseconds since pipeline start.
  final double timestampMs;

  /// Whether active mouth/speech movement is currently detected by VAD.
  final bool isSpeechActive;

  /// Face detection status.
  final bool faceDetected;

  const SubtitleEvent({
    required this.rawVisemes,
    required this.refinedText,
    required this.timestampMs,
    required this.isSpeechActive,
    required this.faceDetected,
  });

  SubtitleEvent copyWith({
    String? rawVisemes,
    String? refinedText,
    double? timestampMs,
    bool? isSpeechActive,
    bool? faceDetected,
  }) {
    return SubtitleEvent(
      rawVisemes: rawVisemes ?? this.rawVisemes,
      refinedText: refinedText ?? this.refinedText,
      timestampMs: timestampMs ?? this.timestampMs,
      isSpeechActive: isSpeechActive ?? this.isSpeechActive,
      faceDetected: faceDetected ?? this.faceDetected,
    );
  }
}
