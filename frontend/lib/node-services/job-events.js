/**
 * Shared Job Event Bus - Plain JS module for cross-boundary sharing
 *
 * This module is intentionally plain JavaScript (not TypeScript) so that
 * both server.js (plain JS) and Next.js API routes (compiled TS) can
 * import the SAME singleton instance.
 *
 * The EventEmitter and jobs Map are shared via globalThis to ensure
 * a single instance across the entire Node.js process.
 */

const { EventEmitter } = require('events');

// Use globalThis to ensure singleton across require() boundaries
if (!globalThis.__jobEvents) {
  const events = new EventEmitter();
  events.setMaxListeners(100);
  globalThis.__jobEvents = events;
}

if (!globalThis.__jobsMap) {
  globalThis.__jobsMap = new Map();
}

module.exports = {
  /** Shared event emitter for job log/status/progress events */
  jobEvents: globalThis.__jobEvents,
  /** Shared jobs map (job ID -> job object) */
  jobsMap: globalThis.__jobsMap,
};
