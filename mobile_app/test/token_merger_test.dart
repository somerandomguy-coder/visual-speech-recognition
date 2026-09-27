import 'package:test/test.dart';
import '../lib/services/vsr_onnx_engine.dart';

void main() {
  group('Sliding Window Token Merger Tests', () {
    test('Stitches exact word overlap without duplication', () {
      const prev = "HELLO HOW";
      const chunk = "HOW ARE YOU";
      final (merged, committed) = VsrOnnxEngine.mergeSlidingTokens(prev, chunk);

      expect(merged, equals("HELLO HOW ARE YOU"));
      expect(committed, contains("HELLO"));
    });

    test('Merges multiple word suffix overlap', () {
      const prev = "WELCOME TO THE FUTURE";
      const chunk = "THE FUTURE OF SPEECH";
      final (merged, committed) = VsrOnnxEngine.mergeSlidingTokens(prev, chunk);

      expect(merged, equals("WELCOME TO THE FUTURE OF SPEECH"));
      expect(committed, contains("WELCOME TO THE"));
    });

    test('Concatenates when no overlap exists', () {
      const prev = "HELLO";
      const chunk = "WORLD";
      final (merged, _) = VsrOnnxEngine.mergeSlidingTokens(prev, chunk);

      expect(merged, equals("HELLO WORLD"));
    });
  });
}
