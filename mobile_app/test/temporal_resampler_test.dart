import 'package:test/test.dart';
import '../lib/services/temporal_resampler.dart';

void main() {
  group('TemporalResampler Tests', () {
    test('Strictly resamples 30 FPS variable input to 25.0 FPS', () {
      final resampler = TemporalResampler<int>(targetFps: 25.0);
      final emitted = <int>[];

      // Simulate 30 incoming frames across 1000ms (~33.3ms intervals)
      for (int i = 0; i < 30; i++) {
        final ts = i * (1000.0 / 30.0);
        emitted.addAll(resampler.addFrame(i, ts));
      }
      emitted.addAll(resampler.flush());

      expect(emitted.length, equals(25),
          reason: '1000ms at 25.0 FPS must yield exactly 25 frames');
    });

    test('Handles variable frame intervals gracefully', () {
      final resampler = TemporalResampler<int>(targetFps: 25.0);
      final emitted = <int>[];

      // Irregular timestamps: 0ms, 50ms, 60ms, 120ms
      emitted.addAll(resampler.addFrame(1, 0.0));
      emitted.addAll(resampler.addFrame(2, 50.0));
      emitted.addAll(resampler.addFrame(3, 60.0));
      emitted.addAll(resampler.addFrame(4, 120.0));
      emitted.addAll(resampler.flush());

      expect(emitted.isNotEmpty, isTrue);
    });
  });
}
