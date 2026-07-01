// ==========================================
// VISIONAI FRONTEND — TF.js COCO-SSD
// Runs 100% in the browser. No server needed.
// ==========================================

// ---- Global State ----
let cocoModel = null;        // Loaded COCO-SSD model
let modelReady = false;      // Is model loaded?
let currentMode = 'upload';  // 'upload' or 'webcam'
let webcamStream = null;
let webcamActive = false;
let webcamRafId = null;      // requestAnimationFrame id
let currentImage = null;     // Currently uploaded Image object
let currentFrameDetections = [];  // Raw detections from last inference
let currentConfidenceThreshold = 0.45;
let highlightedDetectionIndex = null;
let stats = {
    totalFrames: 0,
    startTime: 0,
    fps: 0,
    lastFrameTime: 0
};

// ---- Canvas ----
const canvas = document.getElementById('viewport-canvas');
const ctx = canvas.getContext('2d');

// ---- DOM References ----
const systemStatusDot   = document.getElementById('system-status-dot');
const systemStatusText  = document.getElementById('system-status-text');
const infoEngine        = document.getElementById('info-engine');
const infoLatency       = document.getElementById('info-latency');
const infoFps           = document.getElementById('info-fps');
const metricCount       = document.getElementById('metric-count');
const metricTime        = document.getElementById('metric-time');
const detectionsList    = document.getElementById('detections-list');
const thresholdVal      = document.getElementById('threshold-val');
const thresholdSlider   = document.getElementById('threshold-slider');

const modeUploadBtn     = document.getElementById('mode-upload');
const modeWebcamBtn     = document.getElementById('mode-webcam');
const webcamControlsDiv = document.getElementById('webcam-controls-div');
const webcamToggleBtn   = document.getElementById('webcam-toggle-btn');
const cameraSelect      = document.getElementById('camera-select');

const dropZone          = document.getElementById('drop-zone');
const fileInput         = document.getElementById('file-input');
const canvasContainer   = document.getElementById('canvas-container');
const loadingOverlay    = document.getElementById('loading-overlay');
const webcamVideo       = document.getElementById('webcam-video');

const modelBanner       = document.getElementById('model-loading-banner');
const modelLoadText     = document.getElementById('model-load-text');
const modelLoadProgress = document.getElementById('model-load-progress');

// ==========================================
// INITIALIZATION
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    setupDragAndDrop();
    loadCocoModel();
});

/**
 * Load COCO-SSD model from CDN via TensorFlow.js.
 * Updates status UI as it loads.
 */
async function loadCocoModel() {
    try {
        modelLoadText.textContent = 'Warming up TensorFlow.js...';
        // Ensure TF backend is ready
        await tf.ready();
        modelLoadText.textContent = 'Downloading COCO-SSD model (~5MB)...';
        modelLoadProgress.textContent = '';

        cocoModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        modelReady = true;

        // Warm-up pass so first inference is fast
        const dummy = tf.zeros([1, 64, 64, 3]);
        await cocoModel.detect(dummy);
        dummy.dispose();

        // Update UI
        systemStatusDot.className = 'status-indicator-dot online';
        systemStatusText.textContent = 'Engine Online';
        infoEngine.textContent = 'COCO-SSD (TF.js)';

        // Hide loading banner smoothly
        modelBanner.classList.add('fade-out');
        setTimeout(() => { modelBanner.style.display = 'none'; }, 600);

        // Enumerate cameras now that we have a warm context
        detectCameras();

    } catch (err) {
        console.error('Failed to load COCO-SSD:', err);
        systemStatusDot.className = 'status-indicator-dot offline';
        systemStatusText.textContent = 'Model Load Failed';
        infoEngine.textContent = 'Error';
        modelLoadText.textContent = '⚠ Failed to load model. Check your internet connection.';
        modelLoadProgress.textContent = '';
    }
}

// ==========================================
// MODE SWITCHING (Upload ↔ Live Webcam)
// ==========================================
function setMode(mode) {
    if (currentMode === mode) return;

    if (currentMode === 'webcam' && webcamActive) {
        stopWebcam();
    }

    currentMode = mode;

    if (mode === 'upload') {
        modeUploadBtn.classList.add('active');
        modeWebcamBtn.classList.remove('active');
        webcamControlsDiv.classList.add('hidden');

        if (!currentImage) {
            dropZone.classList.remove('hidden');
            canvasContainer.classList.add('hidden');
        } else {
            dropZone.classList.add('hidden');
            canvasContainer.classList.remove('hidden');
            redrawViewport();
        }

        infoFps.textContent = '--';
        infoLatency.textContent = '--';
    } else {
        modeUploadBtn.classList.remove('active');
        modeWebcamBtn.classList.add('active');
        webcamControlsDiv.classList.remove('hidden');

        dropZone.classList.add('hidden');
        canvasContainer.classList.remove('hidden');

        canvas.width = 640;
        canvas.height = 480;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.font = '16px Inter';
        ctx.textAlign = 'center';
        ctx.fillText("Webcam inactive. Click 'Start Capture' to launch.", canvas.width / 2, canvas.height / 2);
    }

    clearResults();
}

// ==========================================
// DRAG AND DROP / FILE UPLOAD
// ==========================================
function setupDragAndDrop() {
    ['dragenter', 'dragover'].forEach(evt => {
        dropZone.addEventListener(evt, e => {
            e.preventDefault();
            dropZone.classList.add('hover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropZone.addEventListener(evt, e => {
            e.preventDefault();
            dropZone.classList.remove('hover');
        }, false);
    });

    dropZone.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length > 0) processUploadedFile(files[0]);
    }, false);
}

function triggerFileInput() {
    fileInput.click();
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) processUploadedFile(e.target.files[0]);
}

function processUploadedFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload a valid image file.');
        return;
    }

    showLoading(true);
    const reader = new FileReader();
    reader.onload = event => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            canvas.width  = img.naturalWidth;
            canvas.height = img.naturalHeight;
            ctx.drawImage(img, 0, 0);

            dropZone.classList.add('hidden');
            canvasContainer.classList.remove('hidden');

            runDetectionOnImage(img);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

/**
 * Run COCO-SSD inference on a static HTMLImageElement.
 */
async function runDetectionOnImage(imgElement) {
    if (!modelReady) {
        showLoading(false);
        alert('AI model is still loading. Please wait a moment and try again.');
        return;
    }

    const start = performance.now();
    try {
        const predictions = await cocoModel.detect(imgElement, 20);
        const inferenceMs = Math.round(performance.now() - start);

        // Normalise COCO-SSD output → our internal format
        currentFrameDetections = predictions.map(p => ({
            class:      p.class,
            confidence: p.score,
            box:        [
                Math.round(p.bbox[0]),
                Math.round(p.bbox[1]),
                Math.round(p.bbox[0] + p.bbox[2]),
                Math.round(p.bbox[1] + p.bbox[3])
            ]
        }));

        metricTime.innerHTML = `${inferenceMs}<span class="metric-unit">ms</span>`;
        infoLatency.textContent = `${inferenceMs} ms`;

        showLoading(false);
        redrawViewport();
        updateDetectionsList();
    } catch (err) {
        showLoading(false);
        console.error('Detection error:', err);
        alert('Detection failed: ' + err.message);
    }
}

// ==========================================
// WEBCAM STREAM — FIXED & RUNS IN-BROWSER
// ==========================================
async function detectCameras() {
    try {
        // Prompt permission first so labels are available
        const tempStream = await navigator.mediaDevices.getUserMedia({ video: true });
        tempStream.getTracks().forEach(t => t.stop());

        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(d => d.kind === 'videoinput');

        cameraSelect.innerHTML = '';
        videoDevices.forEach((device, index) => {
            const opt = document.createElement('option');
            opt.value = device.deviceId;
            opt.text  = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(opt);
        });
    } catch (e) {
        console.warn('Could not enumerate cameras:', e);
        const opt = document.createElement('option');
        opt.text = 'Default Camera';
        cameraSelect.appendChild(opt);
    }
}

async function toggleWebcam() {
    if (webcamActive) {
        stopWebcam();
    } else {
        await startWebcam();
    }
}

async function startWebcam() {
    if (!modelReady) {
        alert('AI model is still loading. Please wait and try again.');
        return;
    }

    const deviceId = cameraSelect.value;
    const constraints = {
        video: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            width:  { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
        }
    };

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamVideo.srcObject = webcamStream;

        await new Promise((resolve, reject) => {
            webcamVideo.onloadedmetadata = resolve;
            webcamVideo.onerror = reject;
        });
        await webcamVideo.play();

        webcamActive = true;

        // Sync canvas size to actual video dimensions
        canvas.width  = webcamVideo.videoWidth  || 640;
        canvas.height = webcamVideo.videoHeight || 480;

        webcamToggleBtn.innerHTML = `
            <span class="btn-ripple"></span>
            <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" class="btn-icon"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>
            Stop Capture
        `;
        webcamToggleBtn.classList.add('recording');

        // Reset FPS counters
        stats.startTime    = performance.now();
        stats.totalFrames  = 0;

        // Start the detection loop using rAF for smooth rendering
        webcamDetectionLoop();

    } catch (e) {
        console.error('Webcam error:', e);
        const msg = e.name === 'NotAllowedError'
            ? 'Camera permission denied. Please allow camera access in your browser settings.'
            : 'Could not access webcam: ' + e.message;
        alert(msg);
    }
}

function stopWebcam() {
    // Stop rAF loop
    webcamActive = false;
    if (webcamRafId) {
        cancelAnimationFrame(webcamRafId);
        webcamRafId = null;
    }

    // Release camera tracks
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    webcamVideo.srcObject = null;

    webcamToggleBtn.innerHTML = `
        <span class="btn-ripple"></span>
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" class="btn-icon"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
        Start Capture
    `;
    webcamToggleBtn.classList.remove('recording');

    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#ffffff';
    ctx.font = '16px Inter';
    ctx.textAlign = 'center';
    ctx.fillText('Webcam stopped.', canvas.width / 2, canvas.height / 2);

    infoFps.textContent = '--';
    infoLatency.textContent = '--';
    clearResults();
}

/**
 * Webcam detection loop using requestAnimationFrame + async COCO-SSD.
 * Renders each video frame to canvas, runs inference, draws boxes.
 * Uses a "busy flag" to avoid stacking concurrent inferences.
 */
let isInferenceBusy = false;

async function webcamDetectionLoop() {
    if (!webcamActive) return;

    // Always draw the current video frame (keeps preview smooth)
    if (webcamVideo.readyState >= webcamVideo.HAVE_ENOUGH_DATA) {
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);

        // Run inference only when previous one is done
        if (!isInferenceBusy && modelReady) {
            isInferenceBusy = true;

            const start = performance.now();
            try {
                // Run detection on the live video element directly (most efficient)
                const predictions = await cocoModel.detect(webcamVideo, 20);
                const inferenceMs = Math.round(performance.now() - start);

                if (webcamActive) { // Guard: might have stopped during await
                    currentFrameDetections = predictions.map(p => ({
                        class:      p.class,
                        confidence: p.score,
                        box:        [
                            Math.round(p.bbox[0]),
                            Math.round(p.bbox[1]),
                            Math.round(p.bbox[0] + p.bbox[2]),
                            Math.round(p.bbox[1] + p.bbox[3])
                        ]
                    }));

                    // Redraw current video frame + boxes
                    ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
                    drawAllBoxes();
                    updateDetectionsList();

                    // Update metrics
                    stats.totalFrames++;
                    const elapsed = (performance.now() - stats.startTime) / 1000;
                    const fps = Math.round(stats.totalFrames / elapsed);
                    infoFps.textContent     = `${fps} FPS`;
                    infoLatency.textContent = `${inferenceMs} ms`;
                    metricTime.innerHTML    = `${inferenceMs}<span class="metric-unit">ms</span>`;
                }
            } catch (err) {
                if (webcamActive) console.warn('Webcam detection error:', err);
            }

            isInferenceBusy = false;
        } else if (webcamActive) {
            // While inference is running, still overlay last known boxes
            drawAllBoxes();
        }
    }

    if (webcamActive) {
        webcamRafId = requestAnimationFrame(webcamDetectionLoop);
    }
}

// ==========================================
// VIEWPORT RENDERING
// ==========================================
function redrawViewport() {
    if (currentMode === 'upload' && currentImage) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(currentImage, 0, 0);
        drawAllBoxes();
    } else if (currentMode === 'webcam' && webcamActive) {
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
        drawAllBoxes();
    }
}

/** Draw all bounding boxes that pass the confidence threshold. */
function drawAllBoxes() {
    let renderedCount = 0;
    currentFrameDetections.forEach((det, idx) => {
        if (det.confidence >= currentConfidenceThreshold) {
            renderedCount++;
            drawBoundingBox(det, idx === highlightedDetectionIndex);
        }
    });
    metricCount.textContent = renderedCount;
}

/** Glassmorphic neon bounding box with label tab. */
function drawBoundingBox(detection, isHighlighted) {
    const [x1, y1, x2, y2] = detection.box;
    const w = x2 - x1;
    const h = y2 - y1;

    ctx.save();

    if (isHighlighted) {
        ctx.strokeStyle = 'rgba(255, 0, 127, 0.95)';
        ctx.shadowColor = 'rgba(255, 0, 127, 0.6)';
        ctx.shadowBlur  = 15;
        ctx.lineWidth   = 4;
    } else {
        ctx.strokeStyle = 'rgba(0, 242, 254, 0.85)';
        ctx.shadowColor = 'rgba(0, 242, 254, 0.4)';
        ctx.shadowBlur  = 10;
        ctx.lineWidth   = 3;
    }

    const radius = 6;
    ctx.beginPath();
    ctx.moveTo(x1 + radius, y1);
    ctx.lineTo(x1 + w - radius, y1);
    ctx.quadraticCurveTo(x1 + w, y1, x1 + w, y1 + radius);
    ctx.lineTo(x1 + w, y1 + h - radius);
    ctx.quadraticCurveTo(x1 + w, y1 + h, x1 + w - radius, y1 + h);
    ctx.lineTo(x1 + radius, y1 + h);
    ctx.quadraticCurveTo(x1, y1 + h, x1, y1 + h - radius);
    ctx.lineTo(x1, y1 + radius);
    ctx.quadraticCurveTo(x1, y1, x1 + radius, y1);
    ctx.closePath();
    ctx.stroke();

    ctx.shadowBlur = 0;

    const textLabel = `${detection.class.toUpperCase()} ${Math.round(detection.confidence * 100)}%`;
    ctx.font = 'bold 11px Inter, sans-serif';
    const textWidth = ctx.measureText(textLabel).width;
    const tabHeight = 18;
    const tabWidth  = textWidth + 14;

    ctx.fillStyle = isHighlighted ? 'rgba(255, 0, 127, 0.85)' : 'rgba(7, 9, 19, 0.82)';
    const tabX = x1;
    let   tabY = y1 - tabHeight;
    if (tabY < 0) tabY = y1;

    ctx.beginPath();
    ctx.roundRect(tabX, tabY, tabWidth, tabHeight, [4, 4, 0, 0]);
    ctx.fill();

    ctx.fillStyle    = '#ffffff';
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(textLabel, tabX + 7, tabY + tabHeight / 2);

    ctx.restore();
}

// ==========================================
// DETECTIONS SIDEBAR
// ==========================================
function updateDetectionsList() {
    detectionsList.innerHTML = '';

    const filtered = currentFrameDetections
        .map((det, idx) => ({ ...det, originalIndex: idx }))
        .filter(det => det.confidence >= currentConfidenceThreshold);

    if (filtered.length === 0) {
        detectionsList.innerHTML = `
            <div class="no-detections-placeholder">
                No objects above threshold. Try lowering the slider.
            </div>`;
        return;
    }

    filtered.forEach(det => {
        const pct  = Math.round(det.confidence * 100);
        const item = document.createElement('div');
        item.className = 'detection-item';
        if (det.originalIndex === highlightedDetectionIndex) item.classList.add('highlighted');

        const hue = 180 + (pct - 50) * 1.5;
        item.style.borderLeftColor = `hsla(${hue}, 100%, 50%, 0.8)`;

        item.innerHTML = `
            <div class="detection-name-area">
                <span class="detection-name">${det.class}</span>
                <span class="detection-coords">box: [${det.box.join(', ')}]</span>
            </div>
            <div class="detection-pct-area">
                <span class="detection-pct">${pct}%</span>
                <div class="pct-bar-bg">
                    <div class="pct-bar-fill" style="width:${pct}%"></div>
                </div>
            </div>`;

        item.addEventListener('mouseenter', () => {
            highlightedDetectionIndex = det.originalIndex;
            item.classList.add('highlighted');
            redrawViewport();
        });
        item.addEventListener('mouseleave', () => {
            highlightedDetectionIndex = null;
            item.classList.remove('highlighted');
            redrawViewport();
        });

        detectionsList.appendChild(item);
    });
}

// ==========================================
// UTILITIES
// ==========================================
function updateThreshold(val) {
    currentConfidenceThreshold = val / 100;
    thresholdVal.textContent = `${val}%`;
    redrawViewport();
    updateDetectionsList();
}

function showLoading(show) {
    if (show) {
        loadingOverlay.classList.add('active');
    } else {
        loadingOverlay.classList.remove('active');
    }
}

function clearResults() {
    currentFrameDetections  = [];
    highlightedDetectionIndex = null;
    metricCount.textContent  = '0';
    metricTime.innerHTML     = `0<span class="metric-unit">ms</span>`;
    detectionsList.innerHTML = `
        <div class="no-detections-placeholder">
            No objects detected. Upload an image or start webcam to begin.
        </div>`;
}

/**
 * Load the bundled demo image and run detection on it.
 * Path is relative (works on GitHub Pages).
 */
async function loadDemoImage(event) {
    if (event) event.stopPropagation();
    showLoading(true);

    const demoUrl = 'static/workspace_demo.png';
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = async () => {
        currentImage   = img;
        canvas.width   = img.naturalWidth;
        canvas.height  = img.naturalHeight;
        ctx.drawImage(img, 0, 0);

        dropZone.classList.add('hidden');
        canvasContainer.classList.remove('hidden');

        await runDetectionOnImage(img);
    };
    img.onerror = () => {
        showLoading(false);
        alert('Demo image failed to load.');
    };
    img.src = demoUrl;
}
