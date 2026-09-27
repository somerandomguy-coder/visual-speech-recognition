import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime_flutter/onnxruntime_flutter.dart';

/// Standalone on-device ONNX Runtime VSR inference engine.
class VsrOnnxEngine {
  final String modelAssetPath;
  final String vocabAssetPath;
  final int windowFrames;
  final int stepFrames;

  OrtSession? _session;
  List<String> _tokenList = [];
  final List<Uint8List> _frameBuffer = [];
  String _committedText = "";
  String _fullTranscript = "";

  VsrOnnxEngine({
    this.modelAssetPath = 'assets/models/auto_avsr_visual_int8.onnx',
    this.vocabAssetPath = 'assets/tokens/unigram5000_units.txt',
    this.windowFrames = 37, // 1.5s at 25 FPS
    this.stepFrames = 12,   // 0.5s step
  });

  bool get isLoaded => _session != null && _tokenList.isNotEmpty;
  String get fullTranscript => _fullTranscript;
  String get committedText => _committedText;

  /// Load vocabulary and initialize ONNX Runtime session.
  Future<void> initialize() async {
    OrtEnv.instance.init();

    // 1. Load vocabulary units
    final vocabData = await rootBundle.loadString(vocabAssetPath);
    final lines = vocabData.split('\n');
    final units = <String>[];
    for (final line in lines) {
      final trimmed = line.trim();
      if (trimmed.isNotEmpty) {
        units.add(trimmed.split(' ').first);
      }
    }
    _tokenList = ['<blank>', ...units, '<eos>'];

    // 2. Initialize ONNX Runtime Session
    final sessionOptions = OrtSessionOptions()
      ..setIntraOpNumThreads(4)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);

    final rawAsset = await rootBundle.load(modelAssetPath);
    final bytes = rawAsset.buffer.asUint8List();
    _session = OrtSession.fromBuffer(bytes, sessionOptions);
  }

  void reset() {
    _frameBuffer.clear();
    _committedText = "";
    _fullTranscript = "";
  }

  /// Ingest 96x96 grayscale mouth frames, running sliding-window inference when ready.
  Future<String?> addFrames(List<Uint8List> frames96x96) async {
    _frameBuffer.addAll(frames96x96);

    if (_frameBuffer.length >= windowFrames && isLoaded) {
      final window = _frameBuffer.sublist(0, windowFrames);
      // Advance buffer by stepFrames
      if (_frameBuffer.length >= stepFrames) {
        _frameBuffer.removeRange(0, stepFrames);
      } else {
        _frameBuffer.clear();
      }

      final rawChunk = await _inferWindow(window);
      if (rawChunk.isNotEmpty) {
        final merged = mergeSlidingTokens(_fullTranscript, rawChunk);
        _fullTranscript = merged.$1;
        _committedText = merged.$2;
        return _fullTranscript;
      }
    }
    return null;
  }

  /// Center-crop 96x96 to 88x88, normalize to [-1, 1], and execute ONNX session.
  Future<String> _inferWindow(List<Uint8List> windowFrames) async {
    if (_session == null) return "";

    const t = 37;
    const h = 88;
    const w = 88;
    final floatBuffer = Float32List(1 * 1 * t * h * w);

    // Center crop coordinates from 96x96 to 88x88: offset = (96 - 88) / 2 = 4
    const offset = 4;
    var writeIdx = 0;

    for (int frameIdx = 0; frameIdx < t; frameIdx++) {
      final frame = windowFrames[frameIdx];
      for (int y = offset; y < offset + h; y++) {
        for (int x = offset; x < offset + w; x++) {
          final pixelVal = frame[y * 96 + x];
          // Normalization: (val / 255.0 - 0.421) / 0.165
          final norm = ((pixelVal / 255.0) - 0.421) / 0.165;
          floatBuffer[writeIdx++] = norm;
        }
      }
    }

    final inputShape = [1, 1, t, h, w];
    final inputTensor = OrtValueTensor.createTensorWithDataList(floatBuffer, inputShape);

    final inputs = {'video_frames': inputTensor};
    final runOptions = OrtRunOptions();
    final outputs = await _session!.runAsync(runOptions, inputs);

    inputTensor.release();
    runOptions.release();

    if (outputs == null || outputs.isEmpty) return "";

    final logitsTensor = outputs.first as OrtValueTensor;
    final logitsData = logitsTensor.value as List<dynamic>;

    // Greedy CTC decoding
    final decodedText = ctcGreedyDecode(logitsData, _tokenList);

    for (final element in outputs) {
      element?.release();
    }

    return decodedText;
  }

  /// Greedy CTC decoding: argmax -> collapse repeats -> remove blanks/eos -> join tokens.
  static String ctcGreedyDecode(List<dynamic> logits3D, List<String> tokenList) {
    // Expected shape: [1, T, VocabSize]
    if (logits3D.isEmpty) return "";
    final timesteps = logits3D.first as List<dynamic>;

    final bestTokens = <int>[];
    for (final step in timesteps) {
      final stepLogits = step as List<dynamic>;
      var maxVal = double.negativeInfinity;
      var maxIdx = 0;
      for (int i = 0; i < stepLogits.length; i++) {
        final val = (stepLogits[i] as num).toDouble();
        if (val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }
      bestTokens.add(maxIdx);
    }

    // 1. Collapse consecutive duplicates
    final collapsed = <int>[];
    int? prev;
    for (final tok in bestTokens) {
      if (tok != prev) {
        collapsed.add(tok);
        prev = tok;
      }
    }

    // 2. Remove blank (0) and eos (last index)
    final eosIdx = tokenList.length - 1;
    final words = <String>[];
    for (final tok in collapsed) {
      if (tok != 0 && tok != eosIdx && tok < tokenList.length) {
        words.add(tokenList[tok]);
      }
    }

    var text = words.join('');
    text = text.replaceAll(' ', ' ').replaceAll('▁', ' ').trim();
    return text;
  }

  /// Stitch continuous sliding window tokens by finding longest overlapping word suffix.
  static (String, String) mergeSlidingTokens(String prevText, String newChunk) {
    if (prevText.isEmpty) return (newChunk, "");
    if (newChunk.isEmpty) return (prevText, prevText);

    final prevWords = prevText.trim().split(RegExp(r'\s+'));
    final newWords = newChunk.trim().split(RegExp(r'\s+'));

    final maxOverlap = min(prevWords.length, newWords.length);
    var overlapCount = 0;

    for (int k = maxOverlap; k >= 1; k--) {
      final prevSuffix = prevWords.sublist(prevWords.length - k).join(' ');
      final newPrefix = newWords.sublist(0, k).join(' ');
      if (prevSuffix.toUpperCase() == newPrefix.toUpperCase()) {
        overlapCount = k;
        break;
      }
    }

    String merged;
    if (overlapCount > 0) {
      final remainingNew = newWords.sublist(overlapCount);
      merged = [...prevWords, ...remainingNew].join(' ');
    } else {
      merged = '$prevText $newChunk'.trim();
    }

    final mergedWords = merged.split(RegExp(r'\s+'));
    final committed = mergedWords.length > 2
        ? mergedWords.sublist(0, mergedWords.length - 2).join(' ')
        : merged;

    return (merged, committed);
  }
}
