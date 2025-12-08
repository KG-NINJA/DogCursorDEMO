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

        // Load the model. 
        // 'lite_mobilenet_v2' is the default and fastest.
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

    // Inference
    // coco-ssd detect returns array of { bbox: [x, y, width, height], class: "dog", score: 0.99 }
    const predictions = await model.detect(video);

    // Filter for dogs
    const dogPredictions = predictions.filter(p => p.class === 'dog' && p.score > CONFIDENCE_THRESHOLD);

    // UI Updates
    drawBoxes(dogPredictions);
    updateCursor(dogPredictions);

    const endTime = performance.now();
    const time = endTime - startTime;
    fpsSpan.innerText = (1000 / time).toFixed(1);

    requestAnimationFrame(runInferenceLoop);
}

function drawBoxes(predictions) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Determine scale if video size differs from canvas, but here we matched them in setupCamera
    // COCO-SSD returns coords relative to the input image (video).

    predictions.forEach(prediction => {
        const [x, y, w, h] = prediction.bbox;

        ctx.strokeStyle = "#00FF00"; // Green for High Speed
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

    const centerX = x + w / 2;
    const centerY = y + h / 2;

    const perX = (centerX / canvas.width) * 100;
    const perY = (centerY / canvas.height) * 100;

    // Invert X for mirrored feel same as previous version
    cursor.style.left = `${100 - perX}%`;
    cursor.style.top = `${perY}%`;

    // Optional: visual distinction for high speed cursor
    cursor.style.borderColor = "#00FF00";
    cursor.style.backgroundColor = "rgba(0, 255, 0, 0.5)";
}

window.onload = load;
