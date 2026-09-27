import 'dart:math';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'visual_vad.dart';

/// Warps arbitrary angled face/mouth landmarks to canonical 96x96 grayscale mouth crops.
class AffineMouthWarper {
  final double emaAlpha;
  final int targetSize;
  final double targetMouthWidth;

  Point2D? _smoothedLeft;
  Point2D? _smoothedRight;
  Point2D? _smoothedCenter;

  AffineMouthWarper({
    this.emaAlpha = 0.7,
    this.targetSize = 96,
    this.targetMouthWidth = 48.0,
  });

  void reset() {
    _smoothedLeft = null;
    _smoothedRight = null;
    _smoothedCenter = null;
  }

  /// Warp face camera frame to standard 96x96 grayscale mouth crop using similarity transformation.
  Uint8List? warpMouth({
    required img.Image srcImage,
    required Point2D leftCorner,
    required Point2D rightCorner,
    required Point2D mouthCenter,
  }) {
    // 1. Exponential Moving Average (EMA) landmark smoothing
    if (_smoothedLeft == null) {
      _smoothedLeft = leftCorner;
      _smoothedRight = rightCorner;
      _smoothedCenter = mouthCenter;
    } else {
      _smoothedLeft = Point2D(
        emaAlpha * leftCorner.x + (1.0 - emaAlpha) * _smoothedLeft!.x,
        emaAlpha * leftCorner.y + (1.0 - emaAlpha) * _smoothedLeft!.y,
      );
      _smoothedRight = Point2D(
        emaAlpha * rightCorner.x + (1.0 - emaAlpha) * _smoothedRight!.x,
        emaAlpha * rightCorner.y + (1.0 - emaAlpha) * _smoothedRight!.y,
      );
      _smoothedCenter = Point2D(
        emaAlpha * mouthCenter.x + (1.0 - emaAlpha) * _smoothedCenter!.x,
        emaAlpha * mouthCenter.y + (1.0 - emaAlpha) * _smoothedCenter!.y,
      );
    }

    final dx = _smoothedRight!.x - _smoothedLeft!.x;
    final dy = _smoothedRight!.y - _smoothedLeft!.y;
    final currentWidth = sqrt(dx * dx + dy * dy);
    if (currentWidth < 1.0) return null;

    final scale = targetMouthWidth / currentWidth;
    final angleRad = -atan2(dy, dx); // counter-rotation to level horizontal

    final halfTarget = targetSize / 2.0;
    final cosA = cos(angleRad) * scale;
    final sinA = sin(angleRad) * scale;

    final outputBytes = Uint8List(targetSize * targetSize);

    // 2. Inverse mapping with bilinear interpolation
    for (int dstY = 0; dstY < targetSize; dstY++) {
      for (int dstX = 0; dstX < targetSize; dstX++) {
        // Offset from target center
        final ox = dstX - halfTarget;
        final oy = dstY - halfTarget;

        // Inverse rotate & scale back to src image space
        final srcX = _smoothedCenter!.x + (ox * cosA - oy * sinA) / (scale * scale);
        final srcY = _smoothedCenter!.y + (ox * sinA + oy * cosA) / (scale * scale);

        int pixelVal = 0;
        if (srcX >= 0 && srcX < srcImage.width - 1 && srcY >= 0 && srcY < srcImage.height - 1) {
          final pixel = srcImage.getPixel(srcX.round(), srcY.round());
          // Convert to grayscale: 0.299 R + 0.587 G + 0.114 B
          pixelVal = (0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b).round().clamp(0, 255);
        }

        outputBytes[dstY * targetSize + dstX] = pixelVal;
      }
    }

    return outputBytes;
  }
}
