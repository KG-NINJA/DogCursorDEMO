/**
 * Dog Controlled Canvas - Main Logic
 */

// Logic is imported from logic.js (loaded via HTML)
// Tests verified logic.js functionality.

// --- Main Logic ---

// Configuration
const CONFIG = {
    SCORE_THRESHOLD: 0.5,
    MODEL_PATH: './yolov8n.onnx', // Local file to avoid CORS/Fetch errors
    TARGET_CLASS_ID: 16, // COCO index for 'dog'
    INPUT_SIZE: 640
};

const dom = {
    video: document.getElementById('webcam'),
    canvas: document.getElementById('gameCanvas'),
    status: document.getElementById('status'),
    centroid: document.getElementById('centroid'),
    height: document.getElementById('height'),
    debug: document.getElementById('debug')
};

const ctx = dom.canvas.getContext('2d');
let session = null;
let isRunning = false;

// Filters for X and Y movement
const filterX = new OneEuroFilter(1.0, 0.0, 1.0); // MinCutoff, Beta, DCutoff
const filterY = new OneEuroFilter(1.0, 0.0, 1.0);

// Game State
const state = {
    circle: {
        x: 0, // 0-1 normalized
        y: 0.5, // 0-1 normalized (starts middle)
        radius: 50,
        color: '#ff0033'
    },
    canvasSize: { w: 0, h: 0 }
};

// Initialize
async function init() {
    try {
        dom.status.innerText = "Initializing Camera...";
        await setupCamera();

        dom.status.innerText = "Loading Model...";
        await loadModel();

        // Resize canvas to window
        resizeCanvas();
        window.addEventListener('resize', resizeCanvas);

        dom.status.innerText = "Ready. Show me a Dog!";
        isRunning = true;
        requestAnimationFrame(loop);
    } catch (e) {
        dom.status.innerText = `Error: ${e.message}`;
        console.error(e);
    }
}

function resizeCanvas() {
    dom.canvas.width = window.innerWidth;
    dom.canvas.height = window.innerHeight;
    state.canvasSize.w = window.innerWidth;
    state.canvasSize.h = window.innerHeight;

    // Initial position if not set
    if (state.circle.x === 0) state.circle.x = 0.5;
}

async function setupCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Browser API navigator.mediaDevices.getUserMedia not available");
    }
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 },
        audio: false
    });
    dom.video.srcObject = stream;
    return new Promise((resolve) => {
        dom.video.onloadedmetadata = () => resolve();
    });
}

async function loadModel() {
    // onnxruntime-web usage
    // Using wasm backend for standard browser support
    session = await ort.InferenceSession.create(CONFIG.MODEL_PATH, { executionProviders: ['wasm'] });
    console.log("Model loaded:", session);
}

async function loop(timestamp) {
    if (!isRunning) return;

    // 1. Detect
    const boxes = await detectFrame();

    // 2. Logic (Find Dog, Map Coords)
    if (boxes && boxes.length > 0) {
        // Find best dog
        let bestDog = null;
        let maxScore = -1;

        for (const box of boxes) {
            if (box.classId === CONFIG.TARGET_CLASS_ID && box.score > maxScore) {
                maxScore = box.score;
                bestDog = box;
            }
        }

        if (bestDog) {
            updateState(bestDog, timestamp / 1000); // timestamp in seconds
            dom.status.innerText = `Dog Detected! Score: ${bestDog.score.toFixed(2)}`;
        } else {
            dom.status.innerText = "Scanning... (No Dog)";
        }
    }

    // 3. Render
    render();

    requestAnimationFrame(loop);
}

/**
 * Preprocess, Run, Postprocess
 * Returns array of { classId, score, bbox: [x1, y1, x2, y2] (normalized) }
 */
async function detectFrame() {
    if (!session) return [];

    // Preprocess: Resize video frame to 640x640 and normalize
    const [inputTensor, imgScale] = await preprocess(dom.video);

    // Run inference
    const feeds = { images: inputTensor };
    const results = await session.run(feeds);

    // Output depends on model. Typical YOLOv8 output: [1, 84, 8400]
    // 84 = 4 box coords + 80 class scores
    const output = results[Object.keys(results)[0]]; // Get first output

    // Postprocess
    return postprocess(output, imgScale);
}

async function preprocess(video) {
    // Create an offscreen canvas to resize image
    const w = CONFIG.INPUT_SIZE;
    const h = CONFIG.INPUT_SIZE;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');

    // Draw video to canvas (simple resize)
    ctx.drawImage(video, 0, 0, w, h);

    const imageData = ctx.getImageData(0, 0, w, h);
    const { data } = imageData;
    const input = new Float32Array(1 * 3 * w * h);

    // HWC to CHW and normalize 0-255 -> 0.0-1.0
    for (let i = 0; i < w * h; i++) {
        input[i] = data[i * 4] / 255.0;           // R
        input[i + w * h] = data[i * 4 + 1] / 255.0;   // G
        input[i + 2 * w * h] = data[i * 4 + 2] / 255.0; // B
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, w, h]);

    // Scale factor to map back to original video dimensions is not needed if we work in normalized coords
    // But if we wanted pixels: scaleX = video.videoWidth / w
    return [tensor, { w, h }];
}

function postprocess(output, imgScale) {
    // Transpose if needed, or iterate directly.
    // YOLOv8 output shape is [1, 84, 8400].

    const data = output.data;
    const dims = output.dims; // [1, 84, 8400]
    const [batch, rows, cols] = dims;
    // rows=84 (xc, yc, w, h, ...classes), cols=8400 (anchors)

    const boxes = [];

    // Iterate over 8400 anchors
    for (let c = 0; c < cols; c++) {
        // Find max class score
        let maxScore = 0;
        let classId = -1;

        // Classes start at index 4
        for (let r = 4; r < rows; r++) {
            // data layout is likely [1, 84, 8400] flattened. 
            // So data[r * cols + c].
            const val = data[r * cols + c];
            if (val > maxScore) {
                maxScore = val;
                classId = r - 4;
            }
        }

        if (maxScore > CONFIG.SCORE_THRESHOLD) {
            // Get box
            const cx = data[0 * cols + c];
            const cy = data[1 * cols + c];
            const w = data[2 * cols + c];
            const h = data[3 * cols + c];

            // Normalize to 0-1
            const x1 = (cx - w / 2) / CONFIG.INPUT_SIZE;
            const y1 = (cy - h / 2) / CONFIG.INPUT_SIZE;
            const x2 = (cx + w / 2) / CONFIG.INPUT_SIZE;
            const y2 = (cy + h / 2) / CONFIG.INPUT_SIZE;

            boxes.push({
                classId: classId,
                score: maxScore,
                bbox: [x1, y1, x2, y2]
            });
        }
    }

    // Simple NMS (Non-Max Suppression)
    // For prototype speed, just picking highest score might work, 
    // but better to implement basic NMS or just filter top 1.
    // We only care about the MAIN dog, so simply sorting by score and taking top 1 is enough for 1 dog.
    boxes.sort((a, b) => b.score - a.score);

    return boxes;
}

function updateState(dogStats, timestamp) {
    // 1. Calculate raw values
    const center = getCentroid(dogStats.bbox); // returns {x, y} normalized
    const h = getBBoxHeight(dogStats.bbox);    // normalized

    dom.centroid.innerText = `x: ${center.x.toFixed(2)}, y: ${center.y.toFixed(2)}`;
    dom.height.innerText = h.toFixed(2);

    // 2. Smooth values
    // X is horizontal movement. Input 0 (left) - 1 (right). Output Canvas X.
    // ** Mirror X for intuitive control (Dog moves Left in Cam -> User sees Mirror -> moves Left?)
    // Actually standard webcam is mirrored. If I see myself move right, I go right on screen.
    // Let's assume input X (0..1) maps inverted or direct.
    // Usually webcam is mirrored by CSS transform: scaleX(-1). 
    // If CSS mirrors video, the coords are also mirrored if we drew on top.
    // But we are processing raw frames.
    // Let's stick to: center.x maps to canvas width.
    const smoothedX = filterX.filter(center.x, timestamp);

    // H is depth. Larger H = Closer = Move UP? Or just keep it vertical Y?
    // Requirement: "If bbox height increases (dog approaches), move circle up."
    // Requirement: "If bbox height decreases (dog steps back), move circle down."
    // Map H (0.1 small - 0.9 big) to Y (1.0 bottom - 0.0 top).
    // so H increases -> Y decreases (Up).
    const smoothedH = filterY.filter(h, timestamp);

    // Mapping
    // X: 0..1 -> 0..1 (Inverse? If I step Right, I want ball to go Right. 
    // In camera, taking a step Right (User's right) usually appears on Left of image if NOT mirrored.
    // If Mirrored, it appears Right.
    // Let's assume standard interaction: User acts as a mirror.
    // We will Map 1.0 - smoothedX to flip it just incase, or keep as is. Let's try direct first.
    state.circle.x = 1.0 - smoothedX; // Mirroring for natural feeling

    // Y: Map H to Y. 
    // Small Dog (0.2) -> Down (0.9)
    // Big Dog (0.8) -> Up (0.1)
    state.circle.y = mapRange(smoothedH, 0.2, 0.8, 0.9, 0.1);
}

function render() {
    // Clear
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, state.canvasSize.w, state.canvasSize.h);

    // Draw Circle
    const cx = state.circle.x * state.canvasSize.w;
    const cy = state.circle.y * state.canvasSize.h;

    ctx.beginPath();
    ctx.arc(cx, cy, state.circle.radius, 0, 2 * Math.PI);
    ctx.fillStyle = state.circle.color;
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.stroke();
}

window.onload = init;
