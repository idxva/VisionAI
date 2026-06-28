// ==========================================
// CATEYE FRONTEND CONTROLLER
// ==========================================

// Global state variables
let currentMode = 'upload'; // 'upload' or 'webcam'
let webcamStream = null;
let webcamActive = false;
let webcamIntervalId = null;
let currentImage = null; // Store currently uploaded Image object
let currentFrameDetections = []; // Store raw detection results from server
let currentConfidenceThreshold = 0.45; // 45% default
let highlightedDetectionIndex = null; // Index of detection hovered in list
let stats = {
    totalFrames: 0,
    startTime: 0,
    fps: 0,
    lastFrameTime: 0
};

// Canvas elements
const canvas = document.getElementById('viewport-canvas');
const ctx = canvas.getContext('2d');

// DOM Elements
const systemStatusDot = document.getElementById('system-status-dot');
const systemStatusText = document.getElementById('system-status-text');
const infoEngine = document.getElementById('info-engine');
const infoLatency = document.getElementById('info-latency');
const infoFps = document.getElementById('info-fps');
const metricCount = document.getElementById('metric-count');
const metricTime = document.getElementById('metric-time');
const detectionsList = document.getElementById('detections-list');
const thresholdVal = document.getElementById('threshold-val');
const thresholdSlider = document.getElementById('threshold-slider');

const modeUploadBtn = document.getElementById('mode-upload');
const modeWebcamBtn = document.getElementById('mode-webcam');
const webcamControlsDiv = document.getElementById('webcam-controls-div');
const webcamToggleBtn = document.getElementById('webcam-toggle-btn');
const cameraSelect = document.getElementById('camera-select');

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const canvasContainer = document.getElementById('canvas-container');
const loadingOverlay = document.getElementById('loading-overlay');
const webcamVideo = document.getElementById('webcam-video');

// Document ready initialization
document.addEventListener("DOMContentLoaded", () => {
    checkServerStatus();
    setupDragAndDrop();
    detectCameras();
});

// Check if Python Flask server is active & which model is running
async function checkServerStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        if (data.status === 'online') {
            systemStatusDot.className = 'status-indicator-dot online';
            systemStatusText.textContent = 'Engine Online';
            infoEngine.textContent = data.model;
        } else if (data.status === 'fallback_mode') {
            systemStatusDot.className = 'status-indicator-dot online';
            systemStatusText.textContent = 'Engine (Fallback)';
            infoEngine.textContent = 'Mock (YOLO loading)';
        } else {
            setOfflineStatus(data.error);
        }
    } catch (e) {
        setOfflineStatus(e.message);
    }
}

function setOfflineStatus(err) {
    systemStatusDot.className = 'status-indicator-dot offline';
    systemStatusText.textContent = 'Offline';
    infoEngine.textContent = 'Not Connected';
    console.error("Server connection failed:", err);
}

// ==========================================
// INTERACTIVE MODE SWITCHING
// ==========================================
function setMode(mode) {
    if (currentMode === mode) return;
    
    // Stop active webcam if switching away
    if (currentMode === 'webcam' && webcamActive) {
        stopWebcam();
    }
    
    currentMode = mode;
    
    // UI active buttons styling
    if (mode === 'upload') {
        modeUploadBtn.classList.add('active');
        modeWebcamBtn.classList.remove('active');
        webcamControlsDiv.classList.add('hidden');
        
        // Show upload dropzone, hide canvas if no image
        if (!currentImage) {
            dropZone.classList.remove('hidden');
            canvasContainer.classList.add('hidden');
        } else {
            dropZone.classList.add('hidden');
            canvasContainer.classList.remove('hidden');
            redrawViewport();
        }
        
        // Reset webcam stats
        infoFps.textContent = '--';
        infoLatency.textContent = '--';
    } else {
        modeUploadBtn.classList.remove('active');
        modeWebcamBtn.classList.add('active');
        webcamControlsDiv.classList.remove('hidden');
        
        dropZone.classList.add('hidden');
        canvasContainer.classList.remove('hidden');
        
        // Initialize canvas sizing for webcam
        canvas.width = 640;
        canvas.height = 480;
        ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw instructions text on canvas
        ctx.fillStyle = "#ffffff";
        ctx.font = "16px Inter";
        ctx.textAlign = "center";
        ctx.fillText("Webcam inactive. Click 'Start Capture' to launch.", canvas.width/2, canvas.height/2);
    }
    
    clearResults();
}

// ==========================================
// DRAG AND DROP / FILE UPLOAD
// ==========================================
function setupDragAndDrop() {
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.add('hover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropZone.classList.remove('hover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            processUploadedFile(files[0]);
        }
    }, false);
}

function triggerFileInput() {
    fileInput.click();
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        processUploadedFile(files[0]);
    }
}

function processUploadedFile(file) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload a valid image file.");
        return;
    }

    showLoading(true);
    
    const reader = new FileReader();
    reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            
            // Set canvas size matching the image aspect ratio
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            
            // Draw image initially
            ctx.drawImage(img, 0, 0);
            
            dropZone.classList.add('hidden');
            canvasContainer.classList.remove('hidden');
            
            // Submit image to Flask backend
            sendImageToServer(file);
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

async function sendImageToServer(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        showLoading(false);
        
        if (data.status === 'success') {
            currentFrameDetections = data.detections;
            metricTime.innerHTML = `${data.inference_time_ms}<span class="metric-unit">ms</span>`;
            infoLatency.textContent = `${data.inference_time_ms} ms`;
            
            redrawViewport();
            updateDetectionsList();
        } else {
            alert("Detection Error: " + data.message);
        }
    } catch (e) {
        showLoading(false);
        console.error(e);
        alert("Failed to communicate with the vision server.");
    }
}

// ==========================================
// WEBCAM STREAM HANDLING
// ==========================================
async function detectCameras() {
    try {
        // Request temporary permission to resolve device labels
        await navigator.mediaDevices.getUserMedia({ video: true });
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');
        
        cameraSelect.innerHTML = '';
        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.text = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
        });
    } catch (e) {
        console.warn("Could not enumerate cameras: ", e);
        const option = document.createElement('option');
        option.text = "Default Camera";
        cameraSelect.appendChild(option);
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
    const deviceId = cameraSelect.value;
    const constraints = {
        video: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            width: { ideal: 640 },
            height: { ideal: 480 }
        }
    };

    try {
        webcamStream = await navigator.mediaDevices.getUserMedia(constraints);
        webcamVideo.srcObject = webcamStream;
        webcamVideo.onloadedmetadata = () => {
            webcamVideo.play();
            webcamActive = true;
            webcamToggleBtn.innerHTML = `
                <span class="btn-ripple"></span>
                <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" class="btn-icon"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="9" y1="9" x2="15" y2="15"/><line x1="15" y1="9" x2="9" y2="15"/></svg>
                Stop Capture
            `;
            webcamToggleBtn.classList.add('recording');
            
            // Adjust canvas sizing
            canvas.width = webcamVideo.videoWidth || 640;
            canvas.height = webcamVideo.videoHeight || 480;
            
            // Start detection loop
            stats.startTime = Date.now();
            stats.totalFrames = 0;
            captureWebcamLoop();
        };
    } catch (e) {
        console.error("Error launching webcam:", e);
        alert("Webcam Access Denied or Unavailable: " + e.message);
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
    }
    webcamActive = false;
    webcamVideo.srcObject = null;
    webcamToggleBtn.innerHTML = `
        <span class="btn-ripple"></span>
        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none" class="btn-icon"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>
                Start Capture
    `;
    webcamToggleBtn.classList.remove('recording');
    
    // Clear display
    ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ffffff";
    ctx.font = "16px Inter";
    ctx.textAlign = "center";
    ctx.fillText("Webcam stopped.", canvas.width/2, canvas.height/2);
    
    infoFps.textContent = '--';
    infoLatency.textContent = '--';
    clearResults();
}

// Real-time detection loop
async function captureWebcamLoop() {
    if (!webcamActive) return;

    // 1. Draw current video frame onto canvas
    ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);

    // 2. Extract jpeg frame as base64 string
    const frameData = canvas.toDataURL('image/jpeg', 0.65);
    
    // 3. Send to server
    const startInferenceTime = Date.now();
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: frameData })
        });
        const data = await response.json();
        
        if (data.status === 'success' && webcamActive) {
            currentFrameDetections = data.detections;
            
            // Calculate FPS & metrics
            const duration = Date.now() - startInferenceTime;
            stats.totalFrames++;
            const elapsed = (Date.now() - stats.startTime) / 1000;
            stats.fps = Math.round(stats.totalFrames / elapsed);
            
            infoFps.textContent = `${stats.fps} FPS`;
            infoLatency.textContent = `${duration} ms`;
            metricTime.innerHTML = `${data.inference_time_ms}<span class="metric-unit">ms</span>`;

            // Draw bounding boxes on top of webcam frame
            redrawViewport();
            updateDetectionsList();
        }
    } catch (e) {
        console.warn("Detection error in webcam loop:", e);
    }

    // Schedule next frame in ~100ms (roughly 10 FPS target, comfortable for client-server)
    if (webcamActive) {
        setTimeout(captureWebcamLoop, 80);
    }
}

// ==========================================
// VIEWPORT RENDERING & OVERLAYS
// ==========================================
function redrawViewport() {
    // 1. Draw base frame
    if (currentMode === 'upload' && currentImage) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(currentImage, 0, 0);
    } else if (currentMode === 'webcam' && webcamActive) {
        // Redrawn in loop, but if slider adjustments occur we trigger redraw
        ctx.drawImage(webcamVideo, 0, 0, canvas.width, canvas.height);
    } else {
        return;
    }

    // 2. Draw filtered bounding boxes
    let renderedCount = 0;
    currentFrameDetections.forEach((det, idx) => {
        if (det.confidence >= currentConfidenceThreshold) {
            renderedCount++;
            const isHighlighted = (idx === highlightedDetectionIndex);
            drawBoundingBox(det, isHighlighted);
        }
    });

    metricCount.textContent = renderedCount;
}

// Elegant CSS glassmorphic canvas box overlay
function drawBoundingBox(detection, isHighlighted) {
    const [x1, y1, x2, y2] = detection.box;
    const w = x2 - x1;
    const h = y2 - y1;

    // Glowing Neon Glass borders
    ctx.save();
    
    if (isHighlighted) {
        // Glowing Pink accent on hover
        ctx.strokeStyle = "rgba(255, 0, 127, 0.95)";
        ctx.shadowColor = "rgba(255, 0, 127, 0.6)";
        ctx.shadowBlur = 15;
        ctx.lineWidth = 4;
    } else {
        // Glowing Cyan accent normally
        ctx.strokeStyle = "rgba(0, 242, 254, 0.85)";
        ctx.shadowColor = "rgba(0, 242, 254, 0.4)";
        ctx.shadowBlur = 10;
        ctx.lineWidth = 3;
    }

    // Rounded rectangle box
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

    // Clean drawing settings for Label text
    ctx.shadowBlur = 0; 
    
    // Draw small translucent glass header/tab above or inside box
    const textLabel = `${detection.class.toUpperCase()} ${Math.round(detection.confidence * 100)}%`;
    ctx.font = "bold 11px Inter, sans-serif";
    const textWidth = ctx.measureText(textLabel).width;
    const tabHeight = 18;
    const tabWidth = textWidth + 14;

    // Draw Tab background (glassmorphic dark backdrop)
    ctx.fillStyle = isHighlighted ? "rgba(255, 0, 127, 0.85)" : "rgba(7, 9, 19, 0.8)";
    const tabX = x1;
    let tabY = y1 - tabHeight;
    // Keep tab inside canvas bounds
    if (tabY < 0) tabY = y1;

    ctx.beginPath();
    ctx.roundRect(tabX, tabY, tabWidth, tabHeight, [4, 4, 0, 0]);
    ctx.fill();

    // Draw Text
    ctx.fillStyle = "#ffffff";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(textLabel, tabX + 7, tabY + tabHeight / 2);

    ctx.restore();
}

// ==========================================
// SIDEBAR DETAILS & LIST POPULATION
// ==========================================
function updateDetectionsList() {
    detectionsList.innerHTML = '';
    
    const filtered = currentFrameDetections.map((det, idx) => ({ ...det, originalIndex: idx }))
                                           .filter(det => det.confidence >= currentConfidenceThreshold);

    if (filtered.length === 0) {
        detectionsList.innerHTML = `
            <div class="no-detections-placeholder">
                No objects above threshold detected. Try sliding down the threshold.
            </div>
        `;
        return;
    }

    filtered.forEach(det => {
        const pct = Math.round(det.confidence * 100);
        const item = document.createElement('div');
        item.className = 'detection-item';
        if (det.originalIndex === highlightedDetectionIndex) {
            item.classList.add('highlighted');
        }

        // Apply a glowing gradient border based on score
        const hue = 180 + (pct - 50) * 1.5; // Scale cyan to green/blue
        item.style.borderLeftColor = `hsla(${hue}, 100%, 50%, 0.8)`;

        item.innerHTML = `
            <div class="detection-name-area">
                <span class="detection-name">${det.class}</span>
                <span class="detection-coords">box: [${det.box.join(', ')}]</span>
            </div>
            <div class="detection-pct-area">
                <span class="detection-pct">${pct}%</span>
                <div class="pct-bar-bg">
                    <div class="pct-bar-fill" style="width: ${pct}%"></div>
                </div>
            </div>
        `;

        // Interactive hover binding
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
    currentFrameDetections = [];
    highlightedDetectionIndex = null;
    metricCount.textContent = '0';
    metricTime.innerHTML = `0<span class="metric-unit">ms</span>`;
    
    detectionsList.innerHTML = `
        <div class="no-detections-placeholder">
            No objects detected. Upload an image or start webcam to begin.
        </div>
    `;
}

async function loadDemoImage(event) {
    if (event) {
        event.stopPropagation(); // Prevent triggering file input click on dropzone
    }
    
    showLoading(true);
    
    const demoUrl = '/static/workspace_demo.png';
    const img = new Image();
    img.onload = async () => {
        currentImage = img;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        
        dropZone.classList.add('hidden');
        canvasContainer.classList.remove('hidden');
        
        try {
            const res = await fetch(demoUrl);
            const blob = await res.blob();
            const file = new File([blob], "demo.png", { type: "image/png" });
            await sendImageToServer(file);
        } catch (e) {
            showLoading(false);
            console.error("Error analyzing demo image:", e);
            alert("Failed to analyze demo image.");
        }
    };
    img.onerror = () => {
        showLoading(false);
        alert("Demo image failed to load.");
    };
    img.src = demoUrl;
}
