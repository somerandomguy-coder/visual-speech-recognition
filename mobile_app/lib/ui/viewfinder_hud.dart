import 'package:flutter/material.dart';
import '../models/head_pose.dart';

/// Camera viewfinder overlay with oval face alignment guide and diagnostics badges.
class ViewfinderHud extends StatelessWidget {
  final bool isFaceDetected;
  final HeadPose? headPose;
  final double currentFps;
  final double inferenceLatencyMs;
  final VoidCallback? onSwitchCamera;

  const ViewfinderHud({
    super.key,
    required this.isFaceDetected,
    this.headPose,
    required this.currentFps,
    required this.inferenceLatencyMs,
    this.onSwitchCamera,
  });

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // 1. Center Custom Reticle
        CustomPaint(
          size: Size.infinite,
          painter: _ReticlePainter(
            isFaceDetected: isFaceDetected,
            isWithinAngle: headPose?.isWithinThreshold ?? false,
          ),
        ),

        // 2. Top Status & Diagnostics Header
        SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
            child: Row(
              children: [
                // Live status pill
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.6),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: isFaceDetected ? const Color(0xFF10B981) : const Color(0xFFEF4444),
                      width: 1.5,
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 8,
                        height: 8,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: isFaceDetected ? const Color(0xFF10B981) : const Color(0xFFEF4444),
                        ),
                      ),
                      const SizedBox(width: 6),
                      Text(
                        isFaceDetected ? "FACE LOCKED" : "ALIGN FACE",
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 0.5,
                        ),
                      ),
                    ],
                  ),
                ),

                const Spacer(),

                // Latency Badge
                _buildBadge('${inferenceLatencyMs.toStringAsFixed(0)} ms'),
                const SizedBox(width: 6),

                // FPS Badge
                _buildBadge('${currentFps.toStringAsFixed(0)} FPS'),
                const SizedBox(width: 8),

                // Camera Switch Button
                if (onSwitchCamera != null)
                  IconButton(
                    icon: const Icon(Icons.flip_camera_ios, color: Colors.white),
                    onPressed: onSwitchCamera,
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildBadge(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.5),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Color(0xFF94A3B8),
          fontSize: 11,
          fontFamily: 'monospace',
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

class _ReticlePainter extends CustomPainter {
  final bool isFaceDetected;
  final bool isWithinAngle;

  _ReticlePainter({
    required this.isFaceDetected,
    required this.isWithinAngle,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final centerX = size.width / 2.0;
    final centerY = size.height * 0.42;

    final ovalWidth = size.width * 0.65;
    final ovalHeight = size.height * 0.45;

    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0
      ..color = !isFaceDetected
          ? Colors.white.withOpacity(0.3)
          : isWithinAngle
              ? const Color(0xFF10B981).withOpacity(0.8)
              : const Color(0xFFF59E0B).withOpacity(0.8);

    // Draw Head Oval Guide
    final rect = Rect.fromCenter(
      center: Offset(centerX, centerY),
      width: ovalWidth,
      height: ovalHeight,
    );
    canvas.drawOval(rect, paint);

    // Draw Mouth Reticle Target
    final mouthCenterY = centerY + ovalHeight * 0.28;
    final mouthPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5
      ..color = paint.color;

    final mouthRect = Rect.fromCenter(
      center: Offset(centerX, mouthCenterY),
      width: ovalWidth * 0.45,
      height: ovalHeight * 0.22,
    );
    canvas.drawRRect(RRect.fromRectAndRadius(mouthRect, const Radius.circular(8)), mouthPaint);
  }

  @override
  bool shouldRepaint(covariant _ReticlePainter oldDelegate) {
    return oldDelegate.isFaceDetected != isFaceDetected ||
        oldDelegate.isWithinAngle != isWithinAngle;
  }
}
