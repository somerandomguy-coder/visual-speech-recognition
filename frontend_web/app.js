// Visual Speech Recognition - Real-time WebSocket Client

class VSRClient {
    constructor() {
        this.video = document.getElementById('videoElement');
        this.canvas = document.getElementById('captureCanvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

        // HUD Elements
        this.connectionIndicator = document.getElementById('connectionIndicator');
        this.fpsBadge = document.getElementById('fpsBadge');
        this.latencyBadge = document.getElementById('latencyBadge');
        this.poseBadge = document.getElementById('poseBadge');
        this.faceReticle = document.getElementById('faceReticle');
        this.reticleLabel = document.getElementById('reticleLabel');
        this.rawTextContent = document.getElementById('rawTextContent');
        this.refinedTextContent = document.getElementById('refinedTextContent');
        this.cameraFlipBtn = document.getElementById('cameraFlipBtn');
        this.liveModeBtn = document.getElementById('liveModeBtn');
        this.pushToTalkModeBtn = document.getElementById('pushToTalkModeBtn');
        this.pttButton = document.getElementById('pttButton');

        // State
        this.websocket = null;
        this.currentFacingMode = 'user'; // 'user' or 'environment'
        this.isLiveMode = true;
        this.isRecordingPTT = false;
        this.streamInterval = null;
        this.targetFPS = 25;
        this.frameCounter = 0;
        this.lastFPSCheck = performance.now();
        this.lastFrameSentTime = 0;

        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.startCamera();
        this.connectWebSocket();
        this.startStreamingLoop();
    }

    setupEventListeners() {
        this.cameraFlipBtn.addEventListener('click', () => this.toggleCamera());

        this.liveModeBtn.addEventListener('click', () => {
            this.isLiveMode = true;
            this.liveModeBtn.classList.add('active');
            this.pushToTalkModeBtn.classList.remove('active');
            this.pttButton.style.display = 'none';
        });

        this.pushToTalkModeBtn.addEventListener('click', () => {
            this.isLiveMode = false;
            this.pushToTalkModeBtn.classList.add('active');
            this.liveModeBtn.classList.remove('active');
            this.pttButton.style.display = 'block';
        });

        // Push-to-Talk Event Listeners
        const startPTT = (e) => {
            if (e.cancelable) e.preventDefault();
            this.isRecordingPTT = true;
            this.pttButton.classList.add('recording');
            this.pttButton.querySelector('.btn-text').textContent = 'RELEASE TO FINISH';
        };

        const stopPTT = (e) => {
            if (e.cancelable) e.preventDefault();
            this.isRecordingPTT = false;
            this.pttButton.classList.remove('recording');
            this.pttButton.querySelector('.btn-text').textContent = 'HOLD TO TRANSCRIBE';
        };

        this.pttButton.addEventListener('mousedown', startPTT);
        this.pttButton.addEventListener('mouseup', stopPTT);
        this.pttButton.addEventListener('touchstart', startPTT, { passive: false });
        this.pttButton.addEventListener('touchend', stopPTT, { passive: false });

        // Spacebar PTT support
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !this.isLiveMode && !this.isRecordingPTT) {
                startPTT(e);
            }
        });
        window.addEventListener('keyup', (e) => {
            if (e.code === 'Space' && !this.isLiveMode && this.isRecordingPTT) {
                stopPTT(e);
            }
        });

        // Hide PTT by default in Live Mode
        this.pttButton.style.display = 'none';
    }

    async startCamera() {
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
        }

        const constraints = {
            audio: false,
            video: {
                facingMode: this.currentFacingMode,
                width: { ideal: 640 },
                height: { ideal: 480 },
                frameRate: { ideal: 30 }
            }
        };

        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = stream;
            await this.video.play();

            this.canvas.width = this.video.videoWidth || 640;
            this.canvas.height = this.video.videoHeight || 480;
        } catch (err) {
            console.error('Camera access error:', err);
            this.reticleLabel.textContent = 'Camera permission required';
        }
    }

    async toggleCamera() {
        this.currentFacingMode = this.currentFacingMode === 'user' ? 'environment' : 'user';
        await this.startCamera();
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

        this.websocket = new WebSocket(wsUrl);
        this.websocket.binaryType = 'arraybuffer';

        this.websocket.onopen = () => {
            this.connectionIndicator.classList.add('connected');
        };

        this.websocket.onclose = () => {
            this.connectionIndicator.classList.remove('connected');
            setTimeout(() => this.connectWebSocket(), 1500);
        };

        this.websocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleServerMessage(data);
        };
    }

    handleServerMessage(data) {
        if (data.type === 'subtitle_tier1') {
            // Immediate raw CTC token feedback (<150ms)
            if (data.full_raw) {
                this.rawTextContent.textContent = data.full_raw;
            }
            if (data.latency_ms) {
                this.latencyBadge.textContent = `${Math.round(data.latency_ms)} ms`;
            }
        } else if (data.type === 'subtitle_tier2') {
            // Refined sentence from LLM homophene resolver
            if (data.corrected_text) {
                this.typewriterEffect(data.corrected_text);
            }
        }

        // Face detection & Head pose feedback
        if (data.face_detected !== undefined) {
            if (data.face_detected) {
                this.faceReticle.classList.add('detected');
                this.reticleLabel.textContent = 'Face tracked';
            } else {
                this.faceReticle.classList.remove('detected');
                this.reticleLabel.textContent = 'Align face in frame';
            }
        }

        if (data.head_pose) {
            const [yaw, pitch, _] = data.head_pose;
            this.poseBadge.textContent = `Y:${Math.round(yaw)}° P:${Math.round(pitch)}°`;
        }
    }

    typewriterEffect(text) {
        this.refinedTextContent.textContent = text;
        this.refinedTextContent.style.animation = 'none';
        void this.refinedTextContent.offsetWidth; // trigger reflow
        this.refinedTextContent.style.animation = 'fadeIn 0.3s ease-in-out';
    }

    startStreamingLoop() {
        const intervalMs = 1000 / this.targetFPS; // 40ms for 25 FPS

        setInterval(() => {
            // Send frames if live mode is active, or if user is holding PTT
            const shouldSend = this.isLiveMode || this.isRecordingPTT;
            if (shouldSend && this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                this.captureAndSendFrame();
            }

            // Update FPS diagnostics
            this.frameCounter++;
            const now = performance.now();
            if (now - this.lastFPSCheck >= 1000) {
                const currentFps = Math.round((this.frameCounter * 1000) / (now - this.lastFPSCheck));
                this.fpsBadge.textContent = `${currentFps} FPS`;
                this.frameCounter = 0;
                this.lastFPSCheck = now;
            }
        }, intervalMs);
    }

    captureAndSendFrame() {
        if (!this.video.videoWidth) return;

        if (this.canvas.width !== this.video.videoWidth) {
            this.canvas.width = this.video.videoWidth;
            this.canvas.height = this.video.videoHeight;
        }

        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

        // Convert canvas to compressed JPEG blob (quality 0.7) for low-latency transmission
        this.canvas.toBlob((blob) => {
            if (!blob) return;
            blob.arrayBuffer().then((buffer) => {
                if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
                    this.websocket.send(buffer);
                }
            });
        }, 'image/jpeg', 0.7);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    new VSRClient();
});
