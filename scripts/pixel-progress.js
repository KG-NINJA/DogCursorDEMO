const fs = require('fs');
const path = require('path');

const PLAN_PATH = path.join(__dirname, '../plan.md');
const SVG_PATH = path.join(__dirname, '../progress.svg');

function generatePixelArt() {
    if (!fs.existsSync(PLAN_PATH)) {
        console.error('plan.md not found');
        return;
    }

    const content = fs.readFileSync(PLAN_PATH, 'utf-8');
    const tasks = content.match(/- \[[ x/]?\]/g) || [];
    const total = tasks.length;
    const completed = tasks.filter(t => t.includes('[x]')).length;
    const inProgress = tasks.filter(t => t.includes('[/]')).length;

    if (total === 0) return;

    const percent = Math.round((completed / total) * 100);
    let color = '#dc3545'; // Red
    if (percent >= 50) color = '#28a745'; // Green
    if (percent >= 80) color = '#ffc107'; // Gold

    // Simple strict pixel grid (5x5)
    const size = 5;
    const pixelSize = 20;
    const width = size * pixelSize;
    const height = size * pixelSize;

    const fillCount = Math.floor((percent / 100) * (size * size));

    let rects = '';
    for (let i = 0; i < size * size; i++) {
        const x = (i % size) * pixelSize;
        const y = Math.floor(i / size) * pixelSize;
        const fillColor = i < fillCount ? color : '#e9ecef';
        rects += `<rect x="${x}" y="${y}" width="${pixelSize - 2}" height="${pixelSize - 2}" fill="${fillColor}" rx="2" />\n`;
    }

    const svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">
    ${rects}
    <text x="${width / 2}" y="${height / 2 + 5}" font-family="Arial" font-size="20" font-weight="bold" fill="#333" text-anchor="middle" opacity="0.5">${percent}%</text>
  </svg>`;

    fs.writeFileSync(SVG_PATH, svg);
    console.log(`Progress: ${percent}% (Generated ${SVG_PATH})`);

    // Update text in plan.md
    // const newContent = content.replace(/Total: \d+%/, `Total: ${percent}%`);
    // fs.writeFileSync(PLAN_PATH, newContent);
}

generatePixelArt();
