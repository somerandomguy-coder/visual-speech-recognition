import 'package:flutter/material.dart';

/// Floating dual-tier subtitle HUD widget for real-time mobile display.
class SubtitleOverlay extends StatelessWidget {
  final String rawVisemes;
  final String refinedText;
  final bool isSpeaking;
  final bool isLiveMode;
  final VoidCallback onToggleMode;

  const SubtitleOverlay({
    super.key,
    required this.rawVisemes,
    required this.refinedText,
    required this.isSpeaking,
    required this.isLiveMode,
    required this.onToggleMode,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16.0),
      decoration: BoxDecoration(
        color: const Color(0xDD0F172A), // Dark slate with opacity
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24.0)),
        border: Border(
          top: BorderSide(color: Colors.white.withOpacity(0.1), width: 1.0),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.4),
            blurRadius: 16.0,
            offset: const Offset(0, -4),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Mode switch and VAD indicator bar
          Row(
            children: [
              // VAD Indicator
              Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isSpeaking ? const Color(0xFF38BDF8) : Colors.white24,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                isSpeaking ? "SPEECH DETECTED" : "LISTENING (VISUAL)",
                style: const TextStyle(
                  color: Color(0xFF94A3B8),
                  fontSize: 10,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.0,
                ),
              ),

              const Spacer(),

              // Mode Toggle
              GestureDetector(
                onTap: onToggleMode,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: isLiveMode ? const Color(0xFF2563EB) : Colors.white12,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    isLiveMode ? "CONTINUOUS" : "PUSH TO TALK",
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 12),

          // Tier 2: Refined Natural English Sentence
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            child: Text(
              refinedText.isNotEmpty ? refinedText : "Speak naturally into camera...",
              key: ValueKey<String>(refinedText),
              style: TextStyle(
                color: refinedText.isNotEmpty ? Colors.white : Colors.white38,
                fontSize: 18,
                fontWeight: FontWeight.w600,
                height: 1.3,
              ),
            ),
          ),

          const SizedBox(height: 8),

          // Tier 1: Instant Raw CTC Tokens (<100ms)
          if (rawVisemes.isNotEmpty)
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                  decoration: BoxDecoration(
                    color: const Color(0xFF6366F1).withOpacity(0.3),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: const Text(
                    "RAW",
                    style: TextStyle(
                      color: Color(0xFFA5B4FC),
                      fontSize: 9,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    rawVisemes,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      color: Color(0xFF94A3B8),
                      fontSize: 12,
                      fontFamily: 'monospace',
                    ),
                  ),
                ),
              ],
            ),
        ],
      ),
    );
  }
}
