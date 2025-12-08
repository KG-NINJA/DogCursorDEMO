/**
 * logic.js
 * Contains pure logic for One-Euro Filter and Coordinate Mapping.
 * Designed to be testable in Node.js and runnable in Browser.
 */

// Simple One-Euro Filter implementation
class OneEuroFilter {
    constructor(minCutoff = 1.0, beta = 0.0, dcutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dcutoff = dcutoff;
        this.x_prev = null;
        this.dx_prev = null;
        this.t_prev = null;
    }

    smoothingFactor(t_e, cutoff) {
        const r = 2 * Math.PI * cutoff * t_e;
        return r / (r + 1);
    }

    exponentialSmoothing(a, x, x_prev) {
        return a * x + (1 - a) * x_prev;
    }

    filter(x, t) {
        if (this.x_prev === null) {
            this.x_prev = x;
            this.dx_prev = 0;
            this.t_prev = t;
            return x;
        }

        const t_e = t - this.t_prev;
        // Avoid division by zero or negative time
        if (t_e <= 0) return this.x_prev;

        const a_d = this.smoothingFactor(t_e, this.dcutoff);
        const dx = (x - this.x_prev) / t_e;
        const dx_hat = this.exponentialSmoothing(a_d, dx, this.dx_prev);

        const cutoff = this.minCutoff + this.beta * Math.abs(dx_hat);
        const a = this.smoothingFactor(t_e, cutoff);
        const x_hat = this.exponentialSmoothing(a, x, this.x_prev);

        this.x_prev = x_hat;
        this.dx_prev = dx_hat;
        this.t_prev = t;

        return x_hat;
    }
}

/**
 * Maps input value from [inMin, inMax] to [outMin, outMax]
 * Clamps the output to [outMin, outMax] logic.
 */
function mapRange(value, inMin, inMax, outMin, outMax) {
    const clamped = Math.max(Math.min(value, inMax), inMin);
    // Normalize to 0-1
    const p = (clamped - inMin) / (inMax - inMin);
    // Map to output
    return outMin + p * (outMax - outMin);
}

/**
 * Calculates centroid from normalized bounding box [x1, y1, x2, y2]
 * returns {x, y}
 */
function getCentroid(bbox) {
    const [x1, y1, x2, y2] = bbox;
    return {
        x: (x1 + x2) / 2,
        y: (y1 + y2) / 2
    };
}

/**
 * Calculates height of normalized bounding box
 */
function getBBoxHeight(bbox) {
    const [x1, y1, x2, y2] = bbox;
    return Math.abs(y2 - y1);
}

// Export for Node.js environment
if (typeof module !== 'undefined') {
    module.exports = {
        OneEuroFilter,
        mapRange,
        getCentroid,
        getBBoxHeight
    };
}
