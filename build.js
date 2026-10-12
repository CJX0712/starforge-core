/*
 * build.js — 生成单文件 index.html（内联 StarForge 内核 + 浏览器自检 UI）
 * 用法: node build.js
 */
const fs = require('fs');
const path = require('path');
const lib = fs.readFileSync(path.join(__dirname, 'starforge.js'), 'utf8');

const html = `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>StarForge · 世界级 AI 内核自检</title>
<style>
  :root{ --bg:#fafafa; --card:#fff; --ink:#1a1a1a; --line:#e5e5e5; --ok:#16a34a; --bad:#dc2626; --accent:#2563eb; }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}
  header{background:linear-gradient(135deg,#0f172a,#1e3a8a);color:#fff;padding:28px 24px}
  header h1{margin:0;font-size:22px;letter-spacing:.5px}
  header p{margin:6px 0 0;opacity:.85;font-size:13px}
  main{max-width:920px;margin:0 auto;padding:24px}
  .summary{display:flex;gap:16px;margin:0 0 20px;flex-wrap:wrap}
  .kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 18px;flex:1;min-width:160px}
  .kpi .n{font-size:26px;font-weight:700}
  .kpi .l{font-size:12px;color:#666;margin-top:2px}
  table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
  th,td{text-align:left;padding:12px 14px;border-bottom:1px solid var(--line);vertical-align:top}
  th{background:#f1f5f9;font-size:13px;color:#334155}
  tr:last-child td{border-bottom:none}
  .badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:12px;font-weight:600}
  .pass{background:#dcfce7;color:var(--ok)}
  .fail{background:#fee2e2;color:var(--bad)}
  code{background:#f1f5f9;padding:1px 6px;border-radius:6px;font-size:13px}
  footer{text-align:center;color:#94a3b8;font-size:12px;padding:24px}
  a{color:var(--accent)}
</style>
</head>
<body>
<header>
  <h1>✦ StarForge — 世界级 AI 内核</h1>
  <p>纯 JS · 零依赖 · 数学可验证 · 作者：晨星 · 浏览器内实时自检</p>
</header>
<main>
  <div class="summary" id="summary"></div>
  <table>
    <thead><tr><th style="width:42px">状态</th><th>检验项</th><th>详情（不变量 / 梯度残差）</th></tr></thead>
    <tbody id="rows"></tbody>
  </table>
  <p style="margin-top:18px;color:#475569;font-size:13px">
    核心验证方法：<b>中心差分梯度检验</b>（数值梯度的黄金标准）逐项对比解析梯度；
    <b>Jacobi 特征值迭代</b>做 SVD 支撑 PCA 正交不变量；<b>logsumexp</b> 保证 softmax 数值稳定。
    所有结论在 Node 无头环境同样可复现（<code>node starforge.test.js</code>）。
  </p>
</main>
<footer>StarForge v1.0.0 · 零重写 · 集大成 · 闭环交付 · 署名晨星</footer>

<script>
/* ===== StarForge 内核（内联，与 starforge.js 同源） ===== */
${lib}
/* ===== 自检 UI ===== */
(function(){
  const r = SF.runDiagnostics();
  const sum = document.getElementById('summary');
  sum.innerHTML =
    '<div class="kpi"><div class="n" style="color:'+(r.allGreen?'var(--ok)':'var(--bad)')+'">'+r.passed+'/'+r.total+'</div><div class="l">自检通过</div></div>'+
    '<div class="kpi"><div class="n">'+SF.version+'</div><div class="l">内核版本</div></div>'+
    '<div class="kpi"><div class="n">'+(r.allGreen?'✅ 全绿':'⚠️ 失败')+'</div><div class="l">交付状态</div></div>';
  const tb = document.getElementById('rows');
  tb.innerHTML = r.checks.map(c =>
    '<tr><td><span class="badge '+(c.pass?'pass':'fail')+'">'+(c.pass?'✅':'⚠️')+'</span></td>'+
    '<td>'+c.name+'</td><td><code>'+c.detail+'</code></td></tr>'
  ).join('');
})();
</script>
</body>
</html>`;

fs.writeFileSync(path.join(__dirname, 'index.html'), html, 'utf8');
console.log('index.html 已生成 (' + html.length + ' 字节)');
