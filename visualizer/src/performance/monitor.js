/**
 * performance/monitor.js
 * ======================
 * High-precision WebGL and animation frame performance telemetry.
 * Tracks FPS, frame times, particle update latencies, and LOD switching overhead.
 */

export class PerfMonitor {
    constructor() {
        this.metrics = {
            fps: { current: 0, avg: 60, min: Infinity, max: 0, samples: [] },
            frameTime: { current: 16.6, avg: 16.6, samples: [] },
            tokenUpdate: { current: 0, avg: 0, samples: [] },
            lodUpdate: { current: 0, avg: 0, samples: [] },
            drawCalls: { current: 0 },
        };
        this._lastFrameTimestamp = performance.now();
    }

    /**
     * Records elapsed frame duration and calculates real-time FPS.
     * @param {number} timestamp - Current high-res timestamp
     */
    tick(timestamp) {
        const delta = timestamp - this._lastFrameTimestamp;
        this._lastFrameTimestamp = timestamp;

        if (delta > 0 && delta < 1000) {
            const fps = 1000 / delta;
            this.metrics.fps.current = fps;
            this.metrics.frameTime.current = delta;

            this._pushSample('fps', fps);
            this._pushSample('frameTime', delta);

            if (fps < this.metrics.fps.min) this.metrics.fps.min = fps;
            if (fps > this.metrics.fps.max) this.metrics.fps.max = fps;
        }

        this._updateDomCounter();
    }

    /**
     * Times a specific execution block (e.g. particle simulation).
     * @param {string} sectionName - Metric key
     * @param {number} elapsedMs - Duration in milliseconds
     */
    recordSection(sectionName, elapsedMs) {
        if (!this.metrics[sectionName]) {
            this.metrics[sectionName] = { current: 0, avg: 0, samples: [] };
        }
        this.metrics[sectionName].current = elapsedMs;
        this._pushSample(sectionName, elapsedMs);
    }

    _pushSample(key, value) {
        const target = this.metrics[key];
        target.samples.push(value);
        if (target.samples.length > 60) {
            target.samples.shift();
        }
        target.avg = target.samples.reduce((a, b) => a + b, 0) / target.samples.length;
    }

    _updateDomCounter() {
        const el = document.getElementById('fps-counter');
        if (el) {
            el.innerText = `${Math.round(this.metrics.fps.current)} FPS`;
        }
    }
}

export const perfMonitor = new PerfMonitor();
