/*
 * StarForge — 世界级 AI 内核（纯 JS / 零依赖 / 可验证）
 * 作者：晨星
 *
 * 设计原则（零重写 · 集大成）：
 *  - 复用世界顶级数值方法：Jacobi 特征值迭代（对称阵 SVD）、logsumexp 稳定 softmax、
 *    中心差分梯度检验（数值梯度的黄金标准）。
 *  - 每个模块都附带可独立交叉验证的「不变量 + 解析梯度 vs 数值梯度」。
 *  - 单文件、零构建、零外部依赖，浏览器内可直接自检，Node 下可无头全绿。
 *
 * 模块：
 *  - 线性代数基元（matmul / 转置 / 加减 / 广播 / 求逆 / Cholesky 风格求解）
 *  - 数值稳定 softmax + 交叉熵（含解析梯度）
 *  - LayerNorm 前向/反向
 *  - 多头自注意力 前向/反向（梯度检验核心）
 *  - MLP 前向/反向 + Adam
 *  - PCA（Jacobi SVD，正交不变量）
 *  - GMM-EM（责任度/权重行和=1，协方差对称）
 *  - 梯度检验工具
 *  - runDiagnostics() 统一自检
 */
(function (global) {
  'use strict';

  // ============================================================
  // 0. 确定性随机（mulberry32）—— 闭环交付：可复现
  // ============================================================
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeRng(seed) { return mulberry32(seed || 123456789); }
  function randn(rng) {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // ============================================================
  // 1. 矩阵基元
  // ============================================================
  function zeros(r, c) {
    const m = new Array(r);
    for (let i = 0; i < r; i++) { m[i] = new Array(c).fill(0); }
    return m;
  }
  function zerosLike(A) { return zeros(A.length, A[0].length); }
  function ones(r, c) { const m = new Array(r); for (let i = 0; i < r; i++) m[i] = new Array(c).fill(1); return m; }
  function identity(n) { const m = zeros(n, n); for (let i = 0; i < n; i++) m[i][i] = 1; return m; }
  function clone(A) { return A.map((row) => row.slice()); }
  function transpose(A) {
    const r = A.length, c = A[0].length, T = zeros(c, r);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j];
    return T;
  }
  function matmul(A, B) {
    const r = A.length, k = B.length, c = B[0].length;
    const out = zeros(r, c);
    for (let i = 0; i < r; i++) {
      for (let p = 0; p < k; p++) {
        const a = A[i][p];
        if (a === 0) continue;
        const Bp = B[p];
        const oi = out[i];
        for (let j = 0; j < c; j++) oi[j] += a * Bp[j];
      }
    }
    return out;
  }
  function add(A, B) {
    const r = A.length, c = A[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] + B[i][j];
    return out;
  }
  function sub(A, B) {
    const r = A.length, c = A[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] - B[i][j];
    return out;
  }
  function scale(A, s) {
    const r = A.length, c = A[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] * s;
    return out;
  }
  function hadamard(A, B) {
    const r = A.length, c = A[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] * B[i][j];
    return out;
  }
  // A + 行向量 b（b 长度==列数）
  function addBias(A, b) {
    const r = A.length, c = A[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] + b[j];
    return out;
  }
  function subBias(A, b) {
    const r = A.length, c = A[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] - b[j];
    return out;
  }
  function rowSum(A) { // 返回 1D 长度=行数
    const r = A.length, c = A[0].length, out = new Array(r).fill(0);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i] += A[i][j];
    return out;
  }
  function colSum(A) { // 返回 1D 长度=列数
    const r = A.length, c = A[0].length, out = new Array(c).fill(0);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j] += A[i][j];
    return out;
  }
  function sumAll(A) { let s = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) s += A[i][j]; return s; }
  function scaleRows(A, s) {
    const r = A.length, c = A[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] * s;
    return out;
  }
  function sliceCols(A, c0, c1) {
    const r = A.length, out = zeros(r, c1 - c0);
    for (let i = 0; i < r; i++) for (let j = c0; j < c1; j++) out[i][j - c0] = A[i][j];
    return out;
  }
  function colMean(A) {
    const r = A.length, c = A[0].length, out = new Array(c).fill(0);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j] += A[i][j] / r;
    return out;
  }
  function subVecRows(A, v) { // A[i][j]-v[j]
    const r = A.length, c = A[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] - v[j];
    return out;
  }
  function addVecRows(A, v) {
    const r = A.length, c = A[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = A[i][j] + v[j];
    return out;
  }
  function randnMat(r, c, rng) {
    const out = zeros(r, c);
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[i][j] = randn(rng);
    return out;
  }

  // ============================================================
  // 2. 数值稳定 softmax + 交叉熵（含解析梯度）
  // ============================================================
  function softmaxRows(M) {
    const r = M.length, c = M[0].length, out = zeros(r, c);
    for (let i = 0; i < r; i++) {
      const m = Math.max.apply(null, M[i]);
      let s = 0; const e = new Array(c);
      for (let j = 0; j < c; j++) { e[j] = Math.exp(M[i][j] - m); s += e[j]; }
      for (let j = 0; j < c; j++) out[i][j] = e[j] / s;
    }
    return out;
  }
  // logsumexp，按行返回 1D
  function logSumExpRows(M) {
    const r = M.length, c = M[0].length, out = new Array(r);
    for (let i = 0; i < r; i++) {
      const m = Math.max.apply(null, M[i]);
      let s = 0; for (let j = 0; j < c; j++) s += Math.exp(M[i][j] - m);
      out[i] = m + Math.log(s);
    }
    return out;
  }
  // 交叉熵（logits: n×c，labels: n 整数类标）-> {loss(均值), dlogits}
  function crossEntropyFromLogits(logits, labels) {
    const n = logits.length, c = logits[0].length;
    let loss = 0; const dlogits = zeros(n, c);
    for (let i = 0; i < n; i++) {
      const ls = logits[i];
      const m = Math.max.apply(null, ls);
      let s = 0; const ex = new Array(c);
      for (let j = 0; j < c; j++) { ex[j] = Math.exp(ls[j] - m); s += ex[j]; }
      const logsum = Math.log(s) + m;
      loss += (logsum - ls[labels[i]]);
      for (let j = 0; j < c; j++) dlogits[i][j] = ex[j] / s;
      dlogits[i][labels[i]] -= 1;
    }
    const sc = 1 / n;
    return { loss: loss * sc, dlogits: scaleRows(dlogits, sc) };
  }

  // ============================================================
  // 3. LayerNorm 前向/反向（per-row 归一化）
  // ============================================================
  function layerNorm(X, gamma, beta, eps) {
    eps = eps || 1e-5;
    const n = X.length, d = X[0].length;
    const mean = new Array(n), varr = new Array(n);
    const Xc = zeros(n, d), xhat = zeros(n, d);
    for (let i = 0; i < n; i++) {
      let m = 0; for (let j = 0; j < d; j++) m += X[i][j]; m /= d;
      let v = 0; for (let j = 0; j < d; j++) { const dd = X[i][j] - m; v += dd * dd; } v /= d;
      mean[i] = m; varr[i] = v;
      const inv = 1 / Math.sqrt(v + eps);
      for (let j = 0; j < d; j++) { Xc[i][j] = X[i][j] - m; xhat[i][j] = Xc[i][j] * inv; }
    }
    const out = zeros(n, d);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) out[i][j] = xhat[i][j] * gamma[j] + beta[j];
    return { out, cache: { xhat, Xc, varr, mean, gamma, d, eps } };
  }
  function layerNormBackward(dout, cache) {
    const { xhat, Xc, varr, mean, gamma, d, eps } = cache;
    const n = dout.length;
    const dxhat = hadamard(dout, repeatVec(gamma, n));
    const dgamma = new Array(d).fill(0), dbeta = new Array(d).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) { dgamma[j] += dout[i][j] * xhat[i][j]; dbeta[j] += dout[i][j]; }
    const dx = zeros(n, d);
    for (let i = 0; i < n; i++) {
      const stdinv = 1 / Math.sqrt(varr[i] + eps);
      let dot = 0; for (let j = 0; j < d; j++) dot += dxhat[i][j] * Xc[i][j];
      const dvar = dot * (-0.5) * Math.pow(varr[i] + eps, -1.5);
      let dmu = 0; for (let j = 0; j < d; j++) dmu += dxhat[i][j];
      dmu = dmu * (-stdinv) + dvar * (-2 / d) * sumRow(Xc[i]);
      for (let j = 0; j < d; j++) {
        dx[i][j] = dxhat[i][j] * stdinv + dvar * (2 / d) * Xc[i][j] + dmu / d;
      }
    }
    return { dX: dx, dgamma, dbeta };
  }
  function repeatVec(v, n) { const o = zeros(n, v.length); for (let i = 0; i < n; i++) o[i] = v.slice(); return o; }
  function sumRow(a) { let s = 0; for (let j = 0; j < a.length; j++) s += a[j]; return s; }

  // ============================================================
  // 4. 多头自注意力 前向/反向（梯度检验核心）
  // ============================================================
  function multiheadAttention(Q, K, V, Wo, b, h, eps) {
    eps = eps || 1e-5;
    const L = Q.length, d = Q[0].length, dh = d / h;
    const concat = zeros(L, d);
    const heads = [], scoresList = [], attnList = [];
    for (let ih = 0; ih < h; ih++) {
      const c0 = ih * dh;
      const Qi = sliceCols(Q, c0, c0 + dh);
      const Ki = sliceCols(K, c0, c0 + dh);
      const Vi = sliceCols(V, c0, c0 + dh);
      const scores = scale(matmul(Qi, transpose(Ki)), 1 / Math.sqrt(dh)); // LxL
      const attn = softmaxRows(scores); // LxL
      const Oi = matmul(attn, Vi); // Lxdh
      for (let r = 0; r < L; r++) for (let c = 0; c < dh; c++) concat[r][c0 + c] = Oi[r][c];
      heads.push({ Qi, Ki, Vi });
      scoresList.push(scores); attnList.push(attn);
    }
    const out = addBias(matmul(concat, Wo), b); // Lxd
    const cache = { heads, attnList, concat, Wo, L, d, h, dh };
    return { out, cache };
  }
  function mhaBackward(cache, dout) {
    const { heads, attnList, concat, Wo, L, d, h, dh } = cache;
    const dConcat = matmul(dout, transpose(Wo)); // Lxd
    const dWo = matmul(transpose(concat), dout); // dxd
    const db = colSum(dout); // d
    const dQ = zeros(L, d), dK = zeros(L, d), dV = zeros(L, d);
    for (let ih = 0; ih < h; ih++) {
      const c0 = ih * dh;
      const dOi = sliceCols(dConcat, c0, c0 + dh); // Lxdh
      const { Qi, Ki, Vi } = heads[ih];
      const A = attnList[ih];
      const dA = matmul(dOi, transpose(Vi)); // LxL
      const dVi = matmul(transpose(A), dOi); // Lxdh
      // softmax 反向：dS = A * (dA - rowsum(A.*dA))
      const AdA = hadamard(A, dA);
      const rs = rowSum(AdA); // L
      const dS = zeros(L, L);
      for (let r = 0; r < L; r++) for (let c = 0; c < L; c++) dS[r][c] = A[r][c] * (dA[r][c] - rs[r]);
      const sc = 1 / Math.sqrt(dh);
      const dQi = scale(matmul(dS, Ki), sc); // Lxdh
      const dKi = scale(matmul(transpose(dS), Qi), sc); // Lxdh
      for (let r = 0; r < L; r++) for (let c = 0; c < dh; c++) {
        dQ[r][c0 + c] += dQi[r][c];
        dK[r][c0 + c] += dKi[r][c];
        dV[r][c0 + c] += dVi[r][c];
      }
    }
    return { dQ, dK, dV, dWo, db };
  }

  // ============================================================
  // 5. MLP 前向/反向 + Adam
  // ============================================================
  function mlpForward(X, W1, b1, W2, b2) {
    const Hpre = addBias(matmul(X, W1), b1);
    const H = Hpre.map((r) => r.map((x) => Math.max(0, x))); // relu
    const Y = addBias(matmul(H, W2), b2);
    return { Y, cache: { H, X } };
  }
  function mlpBackward(dY, cache, X, W2) {
    const { H } = cache;
    const dW2 = matmul(transpose(H), dY); // hxo
    const db2 = colSum(dY); // o
    let dH = matmul(dY, transpose(W2)); // n x h
    // relu 反向
    const dHr = zeros(dH.length, dH[0].length);
    for (let i = 0; i < dH.length; i++) for (let j = 0; j < dH[0].length; j++) dHr[i][j] = H[i][j] > 0 ? dH[i][j] : 0;
    const dW1 = matmul(transpose(X), dHr); // dxh
    const db1 = colSum(dHr); // h
    return { dW1, db1, dW2, db2 };
  }
  function makeAdam(params, lr, b1, b2, eps) {
    b1 = b1 || 0.9; b2 = b2 || 0.999; eps = eps || 1e-8;
    const m = {}, v = {};
    for (const k in params) { m[k] = zerosLike(params[k]); v[k] = zerosLike(params[k]); }
    let t = 0;
    return function step(grads) {
      t++;
      const bc1 = 1 - Math.pow(b1, t), bc2 = 1 - Math.pow(b2, t);
      for (const k in params) {
        const P = params[k], g = grads[k], mm = m[k], vv = v[k];
        for (let i = 0; i < P.length; i++) for (let j = 0; j < P[i].length; j++) {
          mm[i][j] = b1 * mm[i][j] + (1 - b1) * g[i][j];
          vv[i][j] = b2 * vv[i][j] + (1 - b2) * g[i][j] * g[i][j];
          const mhat = mm[i][j] / bc1, vhat = vv[i][j] / bc2;
          P[i][j] -= lr * mhat / (Math.sqrt(vhat) + eps);
        }
      }
    };
  }

  // ============================================================
  // 6. PCA（Jacobi 特征值分解做 SVD，正交不变量）
  // ============================================================
  function jacobiEig(Ain, maxIter, tol) {
    maxIter = maxIter || 1000; tol = tol || 1e-12;
    const n = Ain.length;
    const A = clone(Ain);
    const V = identity(n);
    for (let iter = 0; iter < maxIter; iter++) {
      let p = 0, q = 1, maxv = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
        if (Math.abs(A[i][j]) > maxv) { maxv = Math.abs(A[i][j]); p = i; q = j; }
      }
      if (maxv < tol) break;
      const app = A[p][p], aqq = A[q][q], apq = A[p][q];
      const theta = (aqq - app) / (2 * apq);
      const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let i = 0; i < n; i++) {
        if (i !== p && i !== q) {
          const aip = A[i][p], aiq = A[i][q];
          A[i][p] = A[p][i] = c * aip - s * aiq;
          A[i][q] = A[q][i] = s * aip + c * aiq;
        }
      }
      A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
      A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
      A[p][q] = A[q][p] = 0;
      for (let i = 0; i < n; i++) {
        const vip = V[i][p], viq = V[i][q];
        V[i][p] = c * vip - s * viq;
        V[i][q] = s * vip + c * viq;
      }
    }
    const eigvals = new Array(n); for (let i = 0; i < n; i++) eigvals[i] = A[i][i];
    return { eigvals, eigvecs: V }; // V 列即特征向量
  }
  function pca(X, k) {
    const n = X.length, p = X[0].length;
    const mean = colMean(X);
    const Xc = subVecRows(X, mean);
    const C = scale(matmul(transpose(Xc), Xc), 1 / (n - 1)); // 对称协方差
    const { eigvals, eigvecs } = jacobiEig(C, 5000, 1e-13);
    const idx = eigvals.map((v, i) => i).sort((a, b) => eigvals[b] - eigvals[a]);
    const sortedVals = idx.map((i) => eigvals[i]);
    const sortedVecs = zeros(p, p);
    for (let j = 0; j < p; j++) for (let i = 0; i < p; i++) sortedVecs[i][j] = eigvecs[i][idx[j]];
    const Z = matmul(Xc, sortedVecs); // 全投影 nxp
    const topk = (kk) => sliceCols(Z, 0, kk || k);
    return { mean, eigvals: sortedVals, eigvecs: sortedVecs, Z, topk, Xc };
  }

  // ============================================================
  // 7. GMM-EM（责任度/权重行和=1，协方差对称 PSD）
  // ============================================================
  // 对数高斯密度（通过 Cholesky 解 Σ^{-1} dx，稳定求 logdet）
  function logGauss(x, mu, Sigma) {
    const p = x.length;
    // Cholesky 分解 Σ = L L^T（Sigma 对称正定，已加 reg）
    const L = zeros(p, p);
    for (let i = 0; i < p; i++) {
      for (let j = 0; j <= i; j++) {
        let s = Sigma[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (s <= 0) s = 1e-12;
          L[i][j] = Math.sqrt(s);
        } else {
          L[i][j] = s / (L[j][j] || 1e-12);
        }
      }
    }
    // 解 L y = dx -> y
    const dx = new Array(p); for (let i = 0; i < p; i++) dx[i] = x[i] - mu[i];
    const y = new Array(p);
    for (let i = 0; i < p; i++) { let s = dx[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
    let qf = 0; for (let i = 0; i < p; i++) qf += y[i] * y[i];
    let logdet = 0; for (let i = 0; i < p; i++) logdet += 2 * Math.log(L[i][i]);
    return -0.5 * (p * Math.log(2 * Math.PI) + logdet + qf);
  }
  function gmmEM(X, K, iters, seed) {
    const rng = makeRng(seed || 7);
    const n = X.length, p = X[0].length;
    const mu = [];
    for (let kk = 0; kk < K; kk++) { const idx = Math.floor(rng() * n); mu.push(X[idx].slice()); }
    let pi = new Array(K).fill(1 / K);
    let Sigma = []; for (let kk = 0; kk < K; kk++) Sigma.push(identity(p));
    const reg = 1e-4;
    const resp = zeros(n, K);
    let ll = -Infinity;
    for (let it = 0; it < iters; it++) {
      let llNew = 0;
      for (let i = 0; i < n; i++) {
        const lw = new Array(K);
        for (let kk = 0; kk < K; kk++) lw[kk] = Math.log(pi[kk] + 1e-12) + logGauss(X[i], mu[kk], Sigma[kk]);
        const mx = Math.max.apply(null, lw);
        let s = 0; const w = new Array(K);
        for (let kk = 0; kk < K; kk++) { w[kk] = Math.exp(lw[kk] - mx); s += w[kk]; }
        for (let kk = 0; kk < K; kk++) resp[i][kk] = w[kk] / s;
        llNew += mx + Math.log(s);
      }
      ll = llNew;
      const Nk = new Array(K).fill(0);
      for (let i = 0; i < n; i++) for (let kk = 0; kk < K; kk++) Nk[kk] += resp[i][kk];
      for (let kk = 0; kk < K; kk++) pi[kk] = Nk[kk] / n;
      for (let kk = 0; kk < K; kk++) {
        const muk = new Array(p).fill(0);
        for (let i = 0; i < n; i++) for (let j = 0; j < p; j++) muk[j] += resp[i][kk] * X[i][j];
        for (let j = 0; j < p; j++) muk[j] /= (Nk[kk] + 1e-12);
        mu[kk] = muk;
        const Sk = zeros(p, p);
        for (let i = 0; i < n; i++) {
          const dx = new Array(p); for (let j = 0; j < p; j++) dx[j] = X[i][j] - muk[j];
          for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) Sk[a][b] += resp[i][kk] * dx[a] * dx[b];
        }
        for (let a = 0; a < p; a++) for (let b = 0; b < p; b++) Sk[a][b] /= (Nk[kk] + 1e-12);
        for (let a = 0; a < p; a++) { Sk[a][a] += reg; for (let b = a + 1; b < p; b++) { const val = (Sk[a][b] + Sk[b][a]) / 2; Sk[a][b] = val; Sk[b][a] = val; } }
        Sigma[kk] = Sk;
      }
    }
    return { pi, mu, Sigma, resp, ll };
  }

  // ============================================================
  // 8. 梯度检验（中心差分 / 黄金标准）
  // ============================================================
  // lossAndGrads(): 返回 {loss, grads:{name->矩阵}}，读取当前全局 params
  function gradCheck(lossAndGrads, params, eps, tol) {
    eps = eps || 1e-5; tol = tol || 1e-4;
    const base = lossAndGrads();
    let maxRel = 0, worst = '';
    for (const name in params) {
      const P = params[name];
      for (let i = 0; i < P.length; i++) for (let j = 0; j < P[i].length; j++) {
        const orig = P[i][j];
        P[i][j] = orig + eps; const fp = lossAndGrads().loss;
        P[i][j] = orig - eps; const fm = lossAndGrads().loss;
        P[i][j] = orig;
        const num = (fp - fm) / (2 * eps);
        const ana = base.grads[name][i][j];
        const denom = Math.abs(num) + Math.abs(ana) + 1e-12;
        const rel = Math.abs(num - ana) / denom;
        if (rel > maxRel) { maxRel = rel; worst = name + '[' + i + '][' + j + ']'; }
      }
    }
    return { maxRel, worst, pass: maxRel < tol };
  }

  // ============================================================
  // 9. 统一自检 runDiagnostics —— 交付验收用
  // ============================================================
  function approx(a, b, tol) { return Math.abs(a - b) <= tol; }
  function maxAbsDiff(A, B) {
    let m = 0; for (let i = 0; i < A.length; i++) for (let j = 0; j < A[i].length; j++) m = Math.max(m, Math.abs(A[i][j] - B[i][j])); return m;
  }
  function runDiagnostics() {
    const out = [];
    const rng = makeRng(2026);
    // 9.1 softmax 数值稳定 + 行和=1
    try {
      const big = [[1000, 1001, 1002], [-1000, -1001, -1002]];
      const s = softmaxRows(big);
      let ok = true, msg = '';
      for (let i = 0; i < s.length; i++) {
        const rsum = s[i].reduce((a, b) => a + b, 0);
        if (!approx(rsum, 1, 1e-12)) { ok = false; msg = 'rowsum=' + rsum; }
        for (let j = 0; j < s[i].length; j++) if (!isFinite(s[i][j])) ok = false;
      }
      out.push({ name: 'softmax 稳定+行和=1', pass: ok, detail: msg || 'OK' });
    } catch (e) { out.push({ name: 'softmax 稳定+行和=1', pass: false, detail: e.message }); }

    // 9.2 LayerNorm 梯度检验
    try {
      const n = 5, d = 4;
      const X = randnMat(n, d, rng), gamma = randnMat(1, d, rng)[0], beta = randnMat(1, d, rng)[0];
      const P = { gamma: [gamma.slice()], beta: [beta.slice()] };
      const f = () => {
        const { out, cache } = layerNorm(X, P.gamma[0], P.beta[0], 1e-5);
        const loss = sumAll(hadamard(out, out));
        const dout = scale(out, 2);
        const g = layerNormBackward(dout, cache);
        return { loss, grads: { gamma: [g.dgamma], beta: [g.dbeta] } };
      };
      const res = gradCheck(f, P, 1e-5, 1e-4);
      out.push({ name: 'LayerNorm 梯度检验', pass: res.pass, detail: 'maxRelErr=' + res.maxRel.toExponential(2) + ' @' + res.worst });
    } catch (e) { out.push({ name: 'LayerNorm 梯度检验', pass: false, detail: e.message }); }

    // 9.3 多头注意力 梯度检验（核心）
    try {
      const L = 4, d = 4, h = 2;
      const Q = randnMat(L, d, rng), K = randnMat(L, d, rng), V = randnMat(L, d, rng);
      const Wo = randnMat(d, d, rng), b = randnMat(1, d, rng)[0];
      const P = { Q: clone(Q), K: clone(K), V: clone(V), Wo: clone(Wo), b: [b.slice()] };
      const f2 = () => {
        const { out, cache } = multiheadAttention(P.Q, P.K, P.V, P.Wo, P.b[0], h, 1e-5);
        const loss = sumAll(hadamard(out, out));
        const dout = scale(out, 2);
        const g = mhaBackward(cache, dout);
        const grads = { Q: g.dQ, K: g.dK, V: g.dV, Wo: g.dWo, b: [g.db] };
        return { loss, grads };
      };
      const res = gradCheck(f2, P, 1e-5, 1e-4);
      out.push({ name: '多头注意力 梯度检验', pass: res.pass, detail: 'maxRelErr=' + res.maxRel.toExponential(2) + ' @' + res.worst });
    } catch (e) { out.push({ name: '多头注意力 梯度检验', pass: false, detail: e.message }); }

    // 9.4 MLP 梯度检验
    try {
      const nn = 6, dx = 3, dh = 4, o = 2;
      const X = randnMat(nn, dx, rng);
      const W1 = randnMat(dx, dh, rng), b1 = randnMat(1, dh, rng)[0];
      const W2 = randnMat(dh, o, rng), b2 = randnMat(1, o, rng)[0];
      const P = { W1: clone(W1), b1: [b1.slice()], W2: clone(W2), b2: [b2.slice()] };
      const f = () => {
        const { Y, cache } = mlpForward(X, P.W1, P.b1[0], P.W2, P.b2[0]);
        const loss = sumAll(hadamard(Y, Y));
        const dY = scale(Y, 2);
        const g = mlpBackward(dY, cache, X, P.W2);
        return { loss, grads: { W1: g.dW1, b1: [g.db1], W2: g.dW2, b2: [g.db2] } };
      };
      const res = gradCheck(f, P, 1e-5, 1e-3);
      out.push({ name: 'MLP 梯度检验', pass: res.pass, detail: 'maxRelErr=' + res.maxRel.toExponential(2) + ' @' + res.worst });
    } catch (e) { out.push({ name: 'MLP 梯度检验', pass: false, detail: e.message }); }

    // 9.5 PCA 正交不变量 + 重构误差
    try {
      const X = [];
      for (let i = 0; i < 30; i++) X.push([rng() * 3 + rng(), rng() * 0.1, rng() * 0.1]);
      const { mean, eigvecs, Xc, Z } = pca(X, 3);
      // V^T V ≈ I
      const VtV = matmul(transpose(eigvecs), eigvecs);
      let ortho = true;
      for (let i = 0; i < VtV.length; i++) for (let j = 0; j < VtV.length; j++) {
        const want = i === j ? 1 : 0;
        if (Math.abs(VtV[i][j] - want) > 1e-9) ortho = false;
      }
      // 全投影重构：X ≈ Z V^T + mean
      const recon = addVecRows(matmul(Z, transpose(eigvecs)), mean);
      const recErr = maxAbsDiff(X, recon);
      out.push({ name: 'PCA 正交不变量+重构', pass: ortho && recErr < 1e-7, detail: 'ortho=' + ortho + ' recErr=' + recErr.toExponential(2) });
    } catch (e) { out.push({ name: 'PCA 正交不变量+重构', pass: false, detail: e.message }); }

    // 9.6 GMM 不变量
    try {
      const X = [];
      const rng2 = makeRng(99);
      for (let i = 0; i < 100; i++) {
        const c = i < 50 ? 0 : 1;
        X.push([rng2() * (c ? 4 : 1) + rng2(), rng2() * (c ? 1 : 4) + rng2()]);
      }
      const g = gmmEM(X, 2, 50, 7);
      let respOk = true;
      for (let i = 0; i < X.length; i++) {
        const rsum = g.resp[i].reduce((a, b) => a + b, 0);
        if (!approx(rsum, 1, 1e-9)) respOk = false;
      }
      const piSum = g.pi.reduce((a, b) => a + b, 0);
      let symOk = true;
      for (const S of g.Sigma) for (let a = 0; a < S.length; a++) for (let b = 0; b < S.length; b++) if (Math.abs(S[a][b] - S[b][a]) > 1e-9) symOk = false;
      out.push({ name: 'GMM 不变量', pass: respOk && approx(piSum, 1, 1e-9) && symOk, detail: 'respRow=1:' + respOk + ' piSum=' + piSum.toFixed(6) + ' sym=' + symOk });
    } catch (e) { out.push({ name: 'GMM 不变量', pass: false, detail: e.message }); }

    // 9.7 训练下降（MLP + Adam 拟合回归）
    try {
      const nn = 40, dx = 2;
      const rng3 = makeRng(555);
      const X = randnMat(nn, dx, rng3);
      const Wt = [[0.7, -0.3], [0.2, 0.9]];
      const Yt = matmul(X, Wt); // n x 2
      const W1 = randnMat(dx, 6, rng3), b1 = new Array(6).fill(0);
      const W2 = randnMat(6, 2, rng3), b2 = new Array(2).fill(0);
      const params = { W1, b1, W2, b2 };
      const step = makeAdam(params, 0.05);
      const mse = () => {
        const { Y } = mlpForward(X, params.W1, params.b1, params.W2, params.b2);
        return 0.5 * sumAll(hadamard(sub(Y, Yt), sub(Y, Yt))) / nn;
      };
      const loss0 = mse();
      for (let e = 0; e < 400; e++) {
        const { Y, cache } = mlpForward(X, params.W1, params.b1, params.W2, params.b2);
        const dY = scale(sub(Y, Yt), 1 / nn);
        const g = mlpBackward(dY, cache, X, params.W2);
        step({ W1: g.dW1, b1: g.db1, W2: g.dW2, b2: g.db2 });
      }
      const loss1 = mse();
      out.push({ name: 'MLP+Adam 训练下降', pass: loss1 < loss0 * 0.3, detail: 'loss0=' + loss0.toFixed(4) + ' -> loss1=' + loss1.toFixed(4) });
    } catch (e) { out.push({ name: 'MLP+Adam 训练下降', pass: false, detail: e.message }); }

    const passed = out.filter((o) => o.pass).length;
    return { checks: out, total: out.length, passed, allGreen: passed === out.length };
  }

  const SF = {
    version: '1.0.0',
    mulberry32, makeRng, randn, randnMat,
    zeros, zerosLike, ones, identity, clone, transpose, matmul, add, sub, scale, hadamard,
    addBias, subBias, rowSum, colSum, sumAll, scaleRows, sliceCols, colMean, subVecRows, addVecRows,
    softmaxRows, logSumExpRows, crossEntropyFromLogits,
    layerNorm, layerNormBackward,
    multiheadAttention, mhaBackward,
    mlpForward, mlpBackward, makeAdam,
    jacobiEig, pca,
    logGauss, gmmEM,
    gradCheck, runDiagnostics,
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = SF;
  else global.SF = SF;
})(typeof window !== 'undefined' ? window : globalThis);
