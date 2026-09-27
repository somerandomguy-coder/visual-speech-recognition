import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'models/head_pose.dart';
import 'services/affine_mouth_warper.dart';
import 'services/offline_slm_resolver.dart';
import 'services/temporal_resampler.dart';
import 'services/visual_vad.dart';
import 'services/vsr_onnx_engine.dart';
import 'ui/subtitle_overlay.dart';
import 'ui/viewfinder_hud.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
  ]);
  runApp(const SilentVsrApp());
}

class SilentVsrApp extends StatelessWidget {
  const SilentVsrApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Silent VSR',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: Colors.black,
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF6366F1),
          secondary: Color(0xFF10B981),
        ),
      ),
      home: const SilentVsrHomePage(),
    );
  }
}

class SilentVsrHomePage extends StatefulWidget {
  const SilentVsrHomePage({super.key});

  @override
  State<SilentVsrHomePage> createState() => _SilentVsrHomePageState();
}

class _SilentVsrHomePageState extends State<SilentVsrHomePage> {
  // Pipelines and Services
  final _resampler = TemporalResampler<Uint8List>(targetFps: 25.0);
  final _warper = AffineMouthWarper();
  final _vad = VisualVAD();
  final _vsrEngine = VsrOnnxEngine();
  final _slmResolver = OfflineSlmResolver();

  // State
  bool _isInitialized = false;
  bool _isFaceDetected = false;
  bool _isSpeaking = false;
  bool _isLiveMode = true;
  HeadPose? _headPose;
  double _fps = 0.0;
  double _latencyMs = 0.0;
  String _rawVisemes = "";
  String _refinedText = "";

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  Future<void> _initializeServices() async {
    try {
      await _vsrEngine.initialize();
      await _slmResolver.initialize();
      setState(() {
        _isInitialized = true;
      });
    } catch (e) {
      debugPrint("Service initialization error: $e");
    }
  }

  void _onToggleMode() {
    setState(() {
      _isLiveMode = !_isLiveMode;
    });
  }

  void _onSwitchCamera() {
    // Switch front/back camera
    _warper.reset();
    _resampler.reset();
    _vad.reset();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // 1. Camera Viewfinder Background (or mock canvas)
          Positioned.fill(
            child: Container(
              color: const Color(0xFF0F172A),
              child: const Center(
                child: Icon(
                  Icons.camera_alt_outlined,
                  size: 64,
                  color: Colors.white24,
                ),
              ),
            ),
          ),

          // 2. Viewfinder HUD Reticle & Diagnostics
          Positioned.fill(
            child: ViewfinderHud(
              isFaceDetected: _isFaceDetected,
              headPose: _headPose,
              currentFps: _fps,
              inferenceLatencyMs: _latencyMs,
              onSwitchCamera: _onSwitchCamera,
            ),
          ),

          // 3. Bottom Dual-Tier Subtitles
          Positioned(
            left: 0,
            right: 0,
            bottom: 0,
            child: SubtitleOverlay(
              rawVisemes: _rawVisemes,
              refinedText: _refinedText,
              isSpeaking: _isSpeaking,
              isLiveMode: _isLiveMode,
              onToggleMode: _onToggleMode,
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _slmResolver.dispose();
    super.dispose();
  }
}
