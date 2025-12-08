const logic = require('../logic');
const { OneEuroFilter, mapRange, getCentroid, getBBoxHeight } = logic;

console.log("Running Logic Tests...");
let failed = 0;

function assert(condition, message) {
    if (!condition) {
        console.error(`[FAIL] ${message}`);
        failed++;
    } else {
        console.log(`[PASS] ${message}`);
    }
}

function assertClose(actual, expected, tolerance = 0.001, message) {
    if (Math.abs(actual - expected) > tolerance) {
        console.error(`[FAIL] ${message} - Expected ${expected}, got ${actual}`);
        failed++;
    } else {
        console.log(`[PASS] ${message}`);
    }
}

// Test mapRange
assertClose(mapRange(0.5, 0, 1, 0, 100), 50.0, 0.001, "Map mid value");
assertClose(mapRange(1.5, 0, 1, 0, 100), 100.0, 0.001, "Map range clamp high");
assertClose(mapRange(-0.5, 0, 1, 0, 100), 0.0, 0.001, "Map range clamp low");

// Test getCentroid
// Box: [10, 10, 30, 30] -> Center: 20, 20
const c = getCentroid([10, 10, 30, 30]);
assertClose(c.x, 20, 0.001, "Centroid X");
assertClose(c.y, 20, 0.001, "Centroid Y");

// Test getBBoxHeight
const h = getBBoxHeight([10, 10, 30, 50]);
assertClose(h, 40, 0.001, "BBox Height");

// Test OneEuroFilter
const filter = new OneEuroFilter(1.0, 0.0, 1.0);
const t0 = 0;
const val0 = 10;
const filtered0 = filter.filter(val0, t0);
assertClose(filtered0, 10, 0.001, "Filter initial value pass-through");

const t1 = 0.016; // ~60fps
const val1 = 12; // jump
const filtered1 = filter.filter(val1, t1);
// Should smooth the jump. If no smoothing, it would be 12. With smoothing, it should be less.
console.log(`Filtered value after jump: ${filtered1}`);
assert(filtered1 < 12 && filtered1 > 10, "Filter normal smoothing");


if (failed === 0) {
    console.log("\nALL TESTS PASSED (Green)");
    process.exit(0);
} else {
    console.error(`\n${failed} TESTS FAILED`);
    process.exit(1);
}
