/**
 * Dump a ComfyUI workflow JSON as a readable node list for LLMs.
 *
 * ComfyUI JSON is node-spaghetti — raw format has shuffled IDs,
 * opaque link arrays, and coordinates everywhere. LLMs choke on it.
 * This extracts just the structure: node types, titles, key inputs,
 * and signal flow connections.
 *
 * USAGE:
 *   node tools\dump-workflow.js <path-to-workflow.json>
 *
 * EXAMPLES:
 *   node tools\dump-workflow.js "frontend/lib/comfy/templates/workflows/sdxl-knx-v13pt5.json"
 *   node tools\dump-workflow.js "frontend/lib/comfy/templates/workflows/anima-guy90s-v10.json"
 *
 * Supports both ComfyUI API format ({ nodes: [...], links: [...] })
 * and object-keyed format ({ "nodeId": { class_type, inputs } }).
 */
const fs = require('fs');
const path = require('path');

const filePath = process.argv[2];
if (!filePath) {
  console.error('Usage: node dump-workflow.js <path-to-workflow.json>');
  process.exit(1);
}

const raw = fs.readFileSync(filePath, 'utf-8');
const workflow = JSON.parse(raw);

// ComfyUI API format: { nodes: [...], links: [...] }
// Object format: { "nodeId": { class_type, inputs } }
let nodes;
let links = [];

if (workflow.nodes && Array.isArray(workflow.nodes)) {
  // API format
  nodes = workflow.nodes.map(n => ({
    id: n.id,
    type: n.type,
    title: n.title || n._meta?.title || '',
    inputs: (n.inputs || []).filter(i => !i.link).map(i => `${i.name}${i.widget ? ` (widget)` : ''}`),
    connections: (n.inputs || []).filter(i => i.link).map(i => ({ name: i.name, link: i.link })),
  }));
  links = (workflow.links || []).map(l => ({
    id: l[0],
    fromNode: l[1],
    fromOutput: l[2],
    toNode: l[3],
    toInput: l[4],
  }));
  // Build reverse lookup
  const linkMap = {};
  for (const l of links) {
    linkMap[l.id] = l;
  }
  for (const n of nodes) {
    n.connections = n.connections.map(c => {
      const l = linkMap[c.link];
      return l ? `${c.name} ← #${l.fromNode}.out${l.fromOutput}` : `${c.name} ← (unknown link ${c.link})`;
    });
  }
} else {
  // Object format: { "nodeId": { class_type, inputs } }
  nodes = Object.entries(workflow).map(([id, n]) => ({
    id: parseInt(id),
    type: n.class_type || 'Unknown',
    title: n._meta?.title || '',
    inputs: Object.entries(n.inputs || {})
      .filter(([k, v]) => !Array.isArray(v))
      .map(([k, v]) => `${k}=${JSON.stringify(v)}`),
    connections: Object.entries(n.inputs || {})
      .filter(([k, v]) => Array.isArray(v))
      .map(([k, [fromId, fromOutput]]) => `${k} ← #${fromId}.out${fromOutput}`),
  }));
}

// Filter out Notes
const realNodes = nodes.filter(n => n.type !== 'Note' && n.type !== 'Reroute');

console.log(`Workflow: ${path.basename(filePath)}`);
console.log(`Total nodes: ${nodes.length} (${nodes.length - realNodes.length} Notes/Reroutes hidden)`);
console.log('─'.repeat(60));
console.log();

// Group by type
const byType = {};
for (const n of realNodes) {
  const key = `${n.type}`;
  if (!byType[key]) byType[key] = [];
  byType[key].push(n);
}

for (const [type, instances] of Object.entries(byType).sort((a, b) => b[1].length - a[1].length)) {
  const titles = instances.map(n => n.title ? `"${n.title}"` : '').filter(Boolean);
  const label = titles.length ? `  (${titles.join(', ')})` : '';
  console.log(`  ${type} ×${instances.length}${label}`);

  // Show key inputs for first instance
  const first = instances[0];
  if (first.inputs.length > 0) {
    console.log(`    inputs: ${first.inputs.slice(0, 10).join(', ')}`);
  }
  console.log();
}

// Show connections
const connectedNodes = realNodes.filter(n => n.connections.length > 0);
if (connectedNodes.length > 0) {
  console.log('─'.repeat(60));
  console.log('CONNECTIONS (signal flow):');
  console.log();
  for (const n of connectedNodes) {
    console.log(`  #${n.id} ${n.type} ←`);
    for (const conn of n.connections) {
      console.log(`    ${conn}`);
    }
    console.log();
  }
}
