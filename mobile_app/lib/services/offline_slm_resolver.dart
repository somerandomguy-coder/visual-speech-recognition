import 'dart:io';
import 'package:flutter/services.dart';
import 'package:llama_cpp_dart/llama_cpp_dart.dart';

/// On-device SLM service for homophene resolution and syntactic refinement using llama.cpp.
class OfflineSlmResolver {
  final String modelAssetPath;
  final int contextSize;
  final int threads;

  LlamaProcessor? _processor;
  bool _isInitializing = false;

  OfflineSlmResolver({
    this.modelAssetPath = 'assets/models/qwen2.5-0.5b-instruct-q4_k_m.gguf',
    this.contextSize = 512,
    this.threads = 4,
  });

  bool get isLoaded => _processor != null;

  /// Initialize llama.cpp processor from asset.
  Future<void> initialize() async {
    if (_processor != null || _isInitializing) return;
    _isInitializing = true;

    try {
      // 1. Copy model asset to local application documents directory if needed
      final byteData = await rootBundle.load(modelAssetPath);
      final tempDir = Directory.systemTemp;
      final localModelFile = File('${tempDir.path}/qwen2.5-0.5b.gguf');

      if (!await localModelFile.exists() || await localModelFile.length() != byteData.lengthInBytes) {
        final buffer = byteData.buffer;
        await localModelFile.writeAsBytes(
          buffer.asUint8List(byteData.offsetInBytes, byteData.lengthInBytes),
          flush: true,
        );
      }

      // 2. Initialize llama.cpp model params
      final modelParams = ModelParams()
        ..contextSize = contextSize;

      _processor = LlamaProcessor(
        localModelFile.path,
        modelParams,
      );
    } catch (e) {
      // Fallback if llama_cpp fails to initialize
      _processor = null;
    } finally {
      _isInitializing = false;
    }
  }

  /// Resolve raw visual CTC text to clean, natural English sentence.
  Future<String> resolve(String rawVsrText) async {
    final clean = rawVsrText.trim();
    if (clean.length < 2) return "";

    if (_processor == null) {
      return _formatSentenceCase(clean);
    }

    final prompt = '''<|im_start|>system
You are an expert visual speech corrector for silent lip-reading CTC output. Fix phonetic homophene substitutions (b/p/m, d/t/n, g/k) and punctuation. Output ONLY the corrected English sentence.<|im_end|>
<|im_start|>user
HELLO HOW ARE EOU<|im_end|>
<|im_start|>assistant
Hello, how are you?<|im_end|>
<|im_start|>user
WE ARE COIN TO THE SHOB<|im_end|>
<|im_start|>assistant
We are going to the shop.<|im_end|>
<|im_start|>user
$clean<|im_end|>
<|im_start|>assistant
''';

    try {
      final response = _processor!.process(
        prompt,
        temperature: 0.1,
        maxTokens: 48,
        stopTokens: ['<|im_end|>', '\n'],
      );

      final result = response.trim();
      if (result.isNotEmpty) {
        return result.replaceAll(RegExp(r'^["\']|["\']$'), '').trim();
      }
    } catch (e) {
      // Fallback gracefully on exception
    }

    return _formatSentenceCase(clean);
  }

  String _formatSentenceCase(String text) {
    if (text.isEmpty) return "";
    final lower = text.toLowerCase();
    return lower[0].toUpperCase() + lower.substring(1);
  }

  void dispose() {
    _processor = null;
  }
}
