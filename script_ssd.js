const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const cursor = document.getElementById('cursor');
const statusDiv = document.getElementById('status');
const fpsSpan = document.getElementById('fps');

let model;

// Configuration
const CONFIDENCE_THRESHOLD = 0.5;

async function load() {
    try {
        statusDiv.innerText = "Loading COCO-SSD Model...";
        model = await cocoSsd.load({ base: 'lite_mobilenet_v2' });

        statusDiv.innerText = "Model Loaded. Starting Camera...";
        await setupCamera();

        statusDiv.innerText = "Running Detection (High Speed)...";
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
    if (!model || video.paused || video.ended || video.readyState < 2) {
        requestAnimationFrame(runInferenceLoop);
        return;
    }

    const startTime = performance.now();

    const predictions = await model.detect(video);
    const dogPredictions = predictions.filter(p => p.class === 'dog' && p.score > CONFIDENCE_THRESHOLD);

    drawBoxes(dogPredictions);
    updateCursor(dogPredictions);

    const endTime = performance.now();
    const time = endTime - startTime;
    fpsSpan.innerText = (1000 / time).toFixed(1);

    requestAnimationFrame(runInferenceLoop);
}

function drawBoxes(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    predictions.forEach(prediction => {
        const [x, y, w, h] = prediction.bbox;

        ctx.strokeStyle = "#00FF00";
        ctx.lineWidth = 4;
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = "#00FF00";
        ctx.font = "18px Arial";
        ctx.fillText(`Dog ${(prediction.score * 100).toFixed(1)}%`, x, y > 20 ? y - 5 : y + 20);
    });
}

function updateCursor(predictions) {
    if (predictions.length === 0) return;

    // Find largest box
    const mainBox = predictions.reduce((prev, current) => {
        const areaPrev = prev.bbox[2] * prev.bbox[3];
        const areaCurr = current.bbox[2] * current.bbox[3];
        return (areaPrev > areaCurr) ? prev : current;
    });

    const [x, y, w, h] = mainBox.bbox;

    // Nose Tracking Heuristic
    const nose = getDarkestCentroid(x, y, w, h);

    let targetX, targetY;
    if (nose) {
        targetX = nose.x;
        targetY = nose.y;

        ctx.fillStyle = "red";
        ctx.beginPath();
        ctx.arc(nose.x, nose.y, 5, 0, 2 * Math.PI);
        ctx.fill();
    } else {
        targetX = x + w / 2;
        targetY = y + h / 2;
    }

    const perX = (targetX / canvas.width) * 100;
    const perY = (targetY / canvas.height) * 100;

    cursor.style.left = `${100 - perX}%`;
    cursor.style.top = `${perY}%`;

    cursor.style.borderColor = "#00FF00";
    cursor.style.backgroundColor = "rgba(0, 255, 0, 0.5)";
}

// Find the centroid of the darkest pixels in the ROI
function getDarkestCentroid(bx, by, bw, bh) {
    // Floor coords for usage in canvas API
    bx = Math.floor(bx); by = Math.floor(by); bw = Math.floor(bw); bh = Math.floor(bh);

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
    const step = 4; // Check every 4th pixel for speed

    // 1. Scan for darkest value
    for (let i = 0; i < len; i += 4 * step) {
        const lum = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
        if (lum < minLum) minLum = lum;
    }

    // 2. Threshold
    const threshold = Math.max(minLum + 20, 60);

    // 3. Centroid
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
