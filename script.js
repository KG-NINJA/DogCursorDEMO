const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const cursor = document.getElementById('cursor');
const statusDiv = document.getElementById('status');
const inferenceTimeSpan = document.getElementById('inference-time');
const fpsSpan = document.getElementById('fps');

let modelSession;

// Configuration
const CONFIDENCE_THRESHOLD = 0.5;
const DOG_CLASS_ID = 16; // COCO class ID for 'dog'

// Main Load Function
async function load() {
    try {
        statusDiv.innerText = "Loading Model...";

        // Initialize ONNX Runtime Web
        const options = {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        };

        modelSession = await ort.InferenceSession.create('./yolov8n.onnx', options);
        statusDiv.innerText = "Model Loaded. Starting Camera...";

        await setupCamera();
        statusDiv.innerText = "Running Detection...";

        // Start loop
        requestAnimationFrame(runInferenceLoop);

    } catch (e) {
        console.error(e);
        statusDiv.innerText = "Error: " + e.message;
    }
}

async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false
    });
    video.srcObject = stream;

    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            video.play(); // Explicitly play logic
            resolve();
        };
    });
}

async function runInferenceLoop() {
    // 0. Safety Checks
    if (!modelSession) {
        requestAnimationFrame(runInferenceLoop);
        return;
    }

    // If video is not ready, skip this frame but keep looping
    if (video.paused || video.ended || video.readyState < 2) {
        requestAnimationFrame(runInferenceLoop);
        return;
    }

    const startTime = performance.now();

    try {
        // 1. Preprocess
        const [inputTensor, modelWidth, modelHeight] = await preprocess(video);

        // 2. Inference
        // 'images' is the standard input name for YOLOv8 exported to ONNX
        const feeds = { images: inputTensor };
        const results = await modelSession.run(feeds);

        // 3. Postprocess
        // YOLOv8 output is usually 'output0'
        const output = results.output0;
        const boxes = processOutput(output, modelWidth, modelHeight);

        // 4. Update UI
        drawBoxes(boxes);
        updateCursor(boxes);

        // Cleanup memory
        inputTensor.dispose();

    } catch (e) {
        console.error("Inference Error:", e);
    }

    const endTime = performance.now();
    const time = endTime - startTime;

    // UI Update
    inferenceTimeSpan.innerText = time.toFixed(1);
    fpsSpan.innerText = (1000 / time).toFixed(1);

    // Schedule next frame
    requestAnimationFrame(runInferenceLoop);
}

// Preprocessing: Resize & Normalize
async function preprocess(source) {
    const w = 640;
    const h = 640;

    // Draw video to an offscreen canvas to resize
    const offCanvas = document.createElement('canvas');
    offCanvas.width = w;
    offCanvas.height = h;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(source, 0, 0, w, h);

    const imageData = offCtx.getImageData(0, 0, w, h);
    const { data } = imageData;

    const float32Data = new Float32Array(3 * w * h);

    // HWC to CHW and Normalize (0-1)
    for (let i = 0; i < w * h; i++) {
        const r = data[i * 4 + 0] / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;

        float32Data[i] = r;              // R
        float32Data[i + w * h] = g;      // G
        float32Data[i + 2 * w * h] = b;  // B
    }

    const tensor = new ort.Tensor('float32', float32Data, [1, 3, h, w]);
    return [tensor, w, h];
}

// Postprocessing: Logic to handle YOLOv8 [1, 84, 8400] output
function processOutput(output, modelW, modelH) {
    const data = output.data;
    const [batch, channels, anchors] = output.dims; // [1, 84, 8400]

    let boxes = [];

    // Loop through anchors (8400)
    for (let i = 0; i < anchors; i++) {
        // Find max class score
        let maxScore = 0;
        let maxClass = -1;

        // Channel layout: 0:x, 1:y, 2:w, 3:h, 4..83: classes
        // Stride is 'anchors' (8400) because it's [channels, anchors] flattened?
        // Actually for [1, 84, 8400], data is flat array.
        // Index [0, c, i] = data[c * anchors + i]

        for (let c = 0; c < 80; c++) {
            const prob = data[(4 + c) * anchors + i];
            if (prob > maxScore) {
                maxScore = prob;
                maxClass = c;
            }
        }

        if (maxScore > CONFIDENCE_THRESHOLD && maxClass === DOG_CLASS_ID) {
            const xc = data[0 * anchors + i];
            const yc = data[1 * anchors + i];
            const w = data[2 * anchors + i];
            const h = data[3 * anchors + i];

            const x = xc - w / 2;
            const y = yc - h / 2;

            boxes.push({ x, y, w, h, score: maxScore, class: maxClass });
        }
    }

    return nms(boxes);
}

// Simple NMS
function nms(boxes) {
    if (boxes.length === 0) return [];

    boxes.sort((a, b) => b.score - a.score);

    const result = [];
    while (boxes.length > 0) {
        const best = boxes.shift();
        result.push(best);

        boxes = boxes.filter(b => {
            const iou = calculateIoU(best, b);
            return iou < 0.45;
        });
    }
    return result;
}

function calculateIoU(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = a.w * a.h;
    const areaB = b.w * b.h;

    return intersection / (areaA + areaB - intersection);
}

function drawBoxes(boxes) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 640;

    boxes.forEach(box => {
        const x = box.x * scaleX;
        const y = box.y * scaleY;
        const w = box.w * scaleX;
        const h = box.h * scaleY;

        ctx.strokeStyle = "#FFD700";
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "#FFD700";
        ctx.font = "18px Arial";
        ctx.fillText(`Dog ${(box.score * 100).toFixed(1)}%`, x, y > 20 ? y - 5 : y + 20);
    });
}

function updateCursor(boxes) {
    if (boxes.length === 0) return;

    const mainBox = boxes.reduce((prev, current) => (prev.w * prev.h > current.w * current.h) ? prev : current);

    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 640;

    const centerX = (mainBox.x + mainBox.w / 2) * scaleX;
    const centerY = (mainBox.y + mainBox.h / 2) * scaleY;

    const perX = (centerX / canvas.width) * 100;
    const perY = (centerY / canvas.height) * 100;

    // Invert X because of the mirrored video/canvas (transform: scaleX(-1))
    // Raw detection is from the camera perspective (Right is Left), but visual is mirrored.
    // We want the cursor to follow the visual position.
    cursor.style.left = `${100 - perX}%`;
    cursor.style.top = `${perY}%`;
}

window.onload = load;
