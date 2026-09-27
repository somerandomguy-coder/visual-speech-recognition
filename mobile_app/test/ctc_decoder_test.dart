import 'package:test/test.dart';
import '../lib/services/vsr_onnx_engine.dart';

void main() {
  group('CTC Greedy Decoder Tests', () {
    test('Collapses duplicates and removes blanks correctly', () {
      final tokenList = ['<blank>', 'HELLO', 'WORLD', '<eos>'];

      // Simulated logits: [1, 5, 4]
      // 0: HELLO (1)
      // 1: HELLO (1) -> duplicate collapsed
      // 2: <blank> (0) -> separator
      // 3: HELLO (1) -> distinct hello
      // 4: WORLD (2)
      final logits = [
        [
          [0.1, 0.9, 0.0, 0.0],
          [0.1, 0.8, 0.0, 0.0],
          [0.9, 0.1, 0.0, 0.0],
          [0.0, 0.9, 0.1, 0.0],
          [0.0, 0.0, 0.9, 0.1],
        ]
      ];

      final text = VsrOnnxEngine.ctcGreedyDecode(logits, tokenList);
      expect(text, contains('HELLO'));
      expect(text, contains('WORLD'));
    });
  });
}
