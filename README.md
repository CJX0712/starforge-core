# ✦ StarForge — 世界级 AI 内核

> 纯 JavaScript · 零依赖 · 数学可验证 · 可复现 · 署名 **晨星**

StarForge 是一套**站在世界顶级成果之上**的 AI 数值内核。它不重复造轮子，而是直接复用被工业界与学界反复验证的最优方法，并把它们统一到一个可自检、可验证、可开源交付的单一代码体中。

## 设计原则

| 原则 | 落地 |
|------|------|
| **零重写** | 复用 Jacobi 特征值迭代（对称阵 SVD）、logsumexp 稳定 softmax、中心差分梯度检验等世界顶级数值方法 |
| **集大成** | 每个模块都附带「不变量 + 解析梯度 vs 数值梯度」双重交叉验证 |
| **闭环交付** | 单文件内核 + 单文件 HTML 演示 + Node 无头自检，全部可运行、可复现、可验证 |
| **署名** | 作者：晨星 |

## 模块清单

| 模块 | 顶级方法 | 可验证不变量 |
|------|----------|--------------|
| 线性代数基元 | 经典 O(n³) matmul / 转置 / 广播 | 维度一致性 |
| 数值稳定 softmax + 交叉熵 | **logsumexp** 防溢出 | 行和 = 1；交叉熵解析梯度 = 数值梯度（~1e-9） |
| LayerNorm | per-row 归一化 | 反向梯度经中心差分验证（~1e-10） |
| **多头自注意力** | scaled dot-product（Vaswani et al. 2017） | 反向全程梯度检验（~1e-9） |
| MLP + Adam | 标准前/反向 + 自适应矩估计 | 梯度检验 + 训练 loss 单调下降 |
| PCA | **Jacobi 特征值迭代**做 SVD | 特征向量正交 `VᵀV=I`；重构误差 ~1e-16 |
| GMM-EM | 期望最大化 + Cholesky 稳定似然 | 责任度行和=1；混合权重和=1；协方差对称 |

## 验证总纲

每个算法至少有 1 条**可被独立实现交叉验证的不变量**——解析解 vs 数值解、参照梯度 vs 主梯度、结构约束（正交 / 行和）。`runDiagnostics()` 统一输出 7 项自检：

```
✅ softmax 稳定+行和=1
✅ LayerNorm 梯度检验      maxRelErr=9.73e-11
✅ 多头注意力 梯度检验     maxRelErr=6.43e-9
✅ MLP 梯度检验           maxRelErr=1.04e-9
✅ PCA 正交不变量+重构     recErr=6.66e-16
✅ GMM 不变量             respRow=1 / piSum=1.000000 / sym=true
✅ MLP+Adam 训练下降      loss 9.67 -> 0.0001
```

## 快速开始

```bash
# 无头自检（Node，退出码 0 = 全绿）
node starforge.test.js

# 生成单文件 HTML 演示（内联内核，浏览器打开即可自检）
node build.js
# 用浏览器打开 index.html
```

`index.html` 为**单文件、零外部依赖**，内联了完整内核与自检 UI，打开即显示 ✅/⚠️ 自检表。

## 文件结构

```
starforge/
├── starforge.js        # 内核（单源真相，Node + 浏览器通用，含 runDiagnostics）
├── starforge.test.js   # Node 无头自检（退出码即验收）
├── build.js            # 读取内核生成单文件 index.html
├── index.html          # 单文件演示（自动生成，内联自检）
├── README.md
└── LICENSE
```

## 许可

MIT License — 作者：**晨星**
