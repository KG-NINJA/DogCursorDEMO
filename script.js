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
            video.play();
            resolve();
        };
    });
}

async function runInferenceLoop() {
    if (!modelSession) {
        requestAnimationFrame(runInferenceLoop);
        return;
    }

    if (video.paused || video.ended || video.readyState < 2) {
        requestAnimationFrame(runInferenceLoop);
        return;
    }

    const startTime = performance.now();

    try {
        const [inputTensor, modelWidth, modelHeight] = await preprocess(video);
        const feeds = { images: inputTensor };
        const results = await modelSession.run(feeds);
        const output = results.output0;
        const boxes = processOutput(output, modelWidth, modelHeight);

        drawBoxes(boxes);
        updateCursor(boxes);

        inputTensor.dispose();

    } catch (e) {
        console.error("Inference Error:", e);
    }

    const endTime = performance.now();
    const time = endTime - startTime;
    inferenceTimeSpan.innerText = time.toFixed(1);
    fpsSpan.innerText = (1000 / time).toFixed(1);

    requestAnimationFrame(runInferenceLoop);
}

async function preprocess(source) {
    const w = 640;
    const h = 640;
    const offCanvas = document.createElement('canvas');
    offCanvas.width = w;
    offCanvas.height = h;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(source, 0, 0, w, h);

    const imageData = offCtx.getImageData(0, 0, w, h);
    const { data } = imageData;
    const float32Data = new Float32Array(3 * w * h);

    for (let i = 0; i < w * h; i++) {
        const r = data[i * 4 + 0] / 255.0;
        const g = data[i * 4 + 1] / 255.0;
        const b = data[i * 4 + 2] / 255.0;
        float32Data[i] = r;
        float32Data[i + w * h] = g;
        float32Data[i + 2 * w * h] = b;
    }
    const tensor = new ort.Tensor('float32', float32Data, [1, 3, h, w]);
    return [tensor, w, h];
}

function processOutput(output, modelW, modelH) {
    const data = output.data;
    const [batch, channels, anchors] = output.dims;
    let boxes = [];
    for (let i = 0; i < anchors; i++) {
        let maxScore = 0;
        let maxClass = -1;
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

// Update cursor using nose tracking (Dark Point Heuristic)
function updateCursor(boxes) {
    if (boxes.length === 0) return;

    const mainBox = boxes.reduce((prev, current) => (prev.w * prev.h > current.w * current.h) ? prev : current);

    const scaleX = canvas.width / 640;
    const scaleY = canvas.height / 640;

    const boxX = Math.floor(mainBox.x * scaleX);
    const boxY = Math.floor(mainBox.y * scaleY);
    const boxW = Math.floor(mainBox.w * scaleX);
    const boxH = Math.floor(mainBox.h * scaleY);

    // Try to find the darkest point within the dog box
    const nose = getDarkestCentroid(boxX, boxY, boxW, boxH);

    let targetX, targetY;
    if (nose) {
        targetX = nose.x;
        targetY = nose.y;

        // Debug Visual for Nose
        ctx.fillStyle = "red";
        ctx.beginPath();
        ctx.arc(nose.x, nose.y, 5, 0, 2 * Math.PI);
        ctx.fill();
    } else {
        // Fallback to center
        targetX = (mainBox.x + mainBox.w / 2) * scaleX;
        targetY = (mainBox.y + mainBox.h / 2) * scaleY;
    }

    const perX = (targetX / canvas.width) * 100;
    const perY = (targetY / canvas.height) * 100;

    cursor.style.left = `${100 - perX}%`;
    cursor.style.top = `${perY}%`;
}

// Find the centroid of the darkest pixels in the ROI
function getDarkestCentroid(bx, by, bw, bh) {
    if (bx < 0) bx = 0; if (by < 0) by = 0;
    if (bx + bw > canvas.width) bw = canvas.width - bx;
    if (by + bh > canvas.height) bh = canvas.height - by;
    if (bw <= 0 || bh <= 0) return null;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = bw;
    tempCanvas.height = bh;
    const tempCtx = tempCanvas.getContext('2d');

    tempCtx.drawImage(video, bx, by, bw, bh, 0, 0, bw, bh);

    const imageData = tempCtx.getImageData(0, 0, bw, bh);
    const data = imageData.data;
    const len = data.length;

    let sumX = 0;
    let sumY = 0;
    let count = 0;
    let minLum = 255;
    const step = 4;

    // 1. Scan for darkest value
    for (let i = 0; i < len; i += 4 * step) {
        const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        if (lum < minLum) minLum = lum;
    }

    // 2. Threshold relative to darkest, clamp to ensure valid range
    const threshold = Math.max(minLum + 20, 60);

    // 3. Calculate centroid of dark blob
    for (let y = 0; y < bh; y += step) {
        for (let x = 0; x < bw; x += step) {
            const i = (y * bw + x) * 4;
            const r = data[i]; const g = data[i + 1]; const b = data[i + 2];
            const lum = 0.299 * r + 0.587 * g + 0.114 * b;

            if (lum < threshold) {
                sumX += x;
                sumY += y;
                count++;
            }
        }
    }

    if (count > 0) {
        return { x: bx + (sumX / count), y: by + (sumY / count) };
    }
    return null;
}

window.onload = load;
