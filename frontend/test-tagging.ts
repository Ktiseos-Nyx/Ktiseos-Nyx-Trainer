/**
 * Test Script for Node.js WD14 Tagging Service
 * Run with: npx ts-node test-tagging.ts <image-path>
 * 
 * Tests ONNX Runtime on CPU - no GPU required!
 */

import { taggingService, tagSingleImage } from './lib/node-services/tagging-service';
import path from 'path';

async function testTagging() {
  console.log('🧪 Testing WD14 Tagging Service (Node.js ONNX Runtime)\n');

  // Get image path from command line or use default
  const imagePath = process.argv[2] || '../dataset/test/sample.jpg';
  
  console.log(\`Image: \${imagePath}\`);
  console.log(\`Model: SmilingWolf/wd-vit-large-tagger-v3\`);
  console.log(\`Execution: CPU (no GPU required)\n\`);

  try {
    console.log('⏳ Loading model...');
    const startTime = Date.now();

    const results = await tagSingleImage(
      imagePath,
      'SmilingWolf/wd-vit-large-tagger-v3',
      0.35
    );

    const elapsed = Date.now() - startTime;

    console.log(\`✅ Tagging completed in \${elapsed}ms\n\`);
    console.log(\`Found \${results.length} tags:\n\`);

    // Display top 20 tags
    results.slice(0, 20).forEach((r, i) => {
      const confidence = (r.confidence * 100).toFixed(1);
      console.log(\`  \${i + 1}. \${r.tag} (\${confidence}%)\`);
    });

    if (results.length > 20) {
      console.log(\`  ... and \${results.length - 20} more tags\`);
    }

    console.log(\`\n🎉 Test passed! ONNX Runtime is working on CPU.\`);
  } catch (error) {
    console.error('❌ Test failed:', error);
    console.error('\nPossible issues:');
    console.error('  1. Model not downloaded yet (download via Models page first)');
    console.error('  2. Image file not found');
    console.error('  3. ONNX Runtime installation issue (try: npm install onnxruntime-node)');
    process.exit(1);
  }
}

// Run test
testTagging();
