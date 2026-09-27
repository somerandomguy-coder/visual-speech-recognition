/// Head pose representation (yaw, pitch, roll) in degrees.
class HeadPose {
  final double yaw;
  final double pitch;
  final double roll;

  const HeadPose({
    required this.yaw,
    required this.pitch,
    required this.roll,
  });

  /// Check whether the face orientation is within the ±45° threshold for optimal lip reading.
  bool get isWithinThreshold {
    return yaw.abs() <= 45.0 && pitch.abs() <= 35.0;
  }

  @override
  String toString() => 'HeadPose(yaw: ${yaw.toStringAsFixed(1)}°, pitch: ${pitch.toStringAsFixed(1)}°)';
}
