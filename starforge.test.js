/*
 * StarForge 无头自检（Node）
 * 用法: node starforge.test.js
 * 退出码 0 = 全绿；非 0 = 有失败
 */
const SF = require('./starforge.js');

console.log('StarForge v' + SF.version + ' — 无头自检');
console.log('========================================');

const { checks, total, passed, allGreen } = SF.runDiagnostics();

for (const c of checks) {
  const icon = c.pass ? '✅' : '⚠️';
  console.log(icon + ' ' + c.name + '  —  ' + c.detail);
}

console.log('----------------------------------------');
console.log('结果: ' + passed + ' / ' + total + (allGreen ? '  ALL GREEN ✅' : '  FAILED ⚠️'));

if (!allGreen) process.exit(1);
