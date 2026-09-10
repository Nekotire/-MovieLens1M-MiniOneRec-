# Changes from upstream MiniOneRec

这个仓库的目标不是重写 MiniOneRec，而是把原项目变成一套在 MovieLens-1M 上可以真实跑完、可恢复、可核对结果的生成式推荐实验。

| Area | Upstream / original path | This repository |
| --- | --- | --- |
| Dataset | 以原项目数据与路径为主 | 增加 MovieLens-1M official data adapter |
| Preprocessing | 数据集相关脚本 | iterative user/item 5-core + chronological next-item samples + global temporal split |
| Item text | 原项目 item metadata | MovieLens `title + genres` only |
| Embedding | 固定/数据集相关入口 | Qwen3-Embedding 路径、batch、dtype 可配置，并校验 shape/finite values |
| RQ-VAE | 原训练脚本 | YAML 驱动；真实测试 256/512/1024 三档码本；记录 collision stats |
| SID gate | zero-collision 假设 | `rqvae.max_collisions` 可配置；本次公开实验明确记录 3 collisions |
| SFT | 原 MiniOneRec 多任务 SFT | MovieLens 数据适配、checkpoint resume、bounded checkpoints、tokenizer/SID preflight |
| Constrained decoding | 模型/前缀细节较硬编码 | tokenizer-derived legal SID prefix tree |
| Evaluation | 全量生成评估 | 支持固定 seed 的 sampled temporal Test，也支持 full Test |
| RL | GRPO 路径 | 保留 ranking reward/GRPO；默认关闭，作为后续扩展实验 |
| Storage | 路径较分散 | data/model/cache/checkpoint 统一放 DATA_ROOT，Git 只保存代码和小型实验记录 |

## What was actually verified

2026-09-09 至 2026-09-10 在 NVIDIA A800 80GB 上完成：

```text
MovieLens preprocessing
Item embedding
3 rounds of RQ-VAE/SID tuning
Dataset conversion
2-epoch Qwen2.5-1.5B SFT
10k sampled temporal-test constrained evaluation
```

本次没有执行 RL。真实结果和限制见 `experiments/20260910_sft/`。
