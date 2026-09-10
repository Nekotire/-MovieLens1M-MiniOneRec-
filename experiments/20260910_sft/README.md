# 2026-09-10 SFT 实验记录

这份目录只保存可以公开、可以核对的小型实验产物，不包含数据集、Embedding、checkpoint 或模型权重。

## 数据

- MovieLens-1M
- iterative 5-core
- 6040 users / 3416 items / 999611 interactions
- history 最大长度：10
- 全局时间切分：80% / 10% / 10%
- 训练 / 验证 / 测试样本：794856 / 99357 / 99358

## Semantic ID

Item 文本使用 `title + genres`，经 Qwen3-Embedding-0.6B 得到 1024 维向量，再训练三层 RQ-VAE。

实际依次跑了三组码本：

| 码本 | Best Loss | collision_count | collision_rate |
| --- | ---: | ---: | ---: |
| [256, 256, 256] | 0.96145 | 13 | 0.003806 |
| [512, 512, 512] | 0.61659 | 6 | 0.001756 |
| [1024, 1024, 1024] | **0.39637** | **3** | **0.000878** |

最终 3416 个 Item 对应 3413 个唯一 SID。因为训练成本原因，本次允许 `collision_count <= 3` 后继续 SFT；这不是 zero-collision 实验。

## SFT

- Base model：Qwen2.5-1.5B
- epochs：2
- micro batch：8
- gradient accumulation：16
- effective batch：128
- bf16：true
- gradient checkpointing：true
- Trainer steps：24946
- runtime：60864.79 s（约 16 h 54 min）
- train loss：0.69623

最初使用 micro batch 4 / gradient accumulation 32。A800 80GB 显存余量较大，所以调整到 8 / 16，在不改变 effective batch 的情况下提高吞吐。服务器日志中训练单步耗时从前一版约 3.6 s/it 降到后续约 2.1–2.4 s/it。

## SFT Evaluation

完整 Test 有 99358 条。全量 50-beam 生成成本较高，本次在 temporal Test 内使用固定 `seed=2024` 随机抽取 10000 条，使用 constrained beam search，`num_beams=10`。

| 指标 | 结果 |
| --- | ---: |
| HR@1 | 0.0121 |
| HR@3 | 0.0356 |
| HR@5 | 0.0603 |
| HR@10 | **0.1135** |
| NDCG@1 | 0.0121 |
| NDCG@3 | 0.02558 |
| NDCG@5 | 0.03573 |
| NDCG@10 | **0.05279** |
| invalid_item_count | 0 |

3416 个 Item 中均匀随机取 10 个、单目标命中的期望 HR@10 为 `10/3416 = 0.002927`，约 0.293%。当前 HR@10 为 11.35%，约是该随机期望的 38.8 倍。这个比较只用于 sanity check，不是 learned baseline。

## 没有做的实验

RL / GRPO 本次没有运行，因此仓库没有 RL 指标，也不报告“SFT → RL 提升”。后续设计见 `docs/RL_EXTENSION.md`。
