# RL 扩展实验设计

这部分是后续实验方案，不是当前已完成结果。2026-09-10 的公开实验只做到 SFT + SFT Evaluation。

## 从哪里开始

RL 直接加载：

```text
output/MovieLens1M/sft/final_checkpoint
```

因此只要 SFT checkpoint 还在，就不需要重新训练 2 个 epoch 的 SFT。

## 当前 GRPO 设计

`rl.py` 使用 TRL 的 `GRPOConfig` / 自定义 `ReReTrainer`。推荐模式默认：

```yaml
reward_type: ranking
beam_search: true
sync_ref_model: true
beta: 0.001
num_generations: 8
epochs: 1
learning_rate: 1e-6
```

对每个 prompt 生成一组 completion。`reward_type=ranking` 同时启用：

1. `rule_reward`：completion 与真实 target SID 完全一致时给 1，否则给 0；
2. `ndcg_rule_reward`：在同一组 generation 中加入和排名位置相关的 reward，使组内相对顺序也进入优化信号。

GRPO 根据同一 prompt 下多个 completion 的相对 reward 构造 advantage，再更新当前 policy。`sync_ref_model=true` 和 `beta=0.001` 用于约束 policy 不要在 RL 阶段偏离 SFT 模型过快。

## RL 数据不是简单等于 max_samples

`max_samples` 先对原始训练 CSV 做固定 seed 的 reservoir sampling，然后代码会构造多种 RL prompt：

- sequential recommendation prompt；
- item title → SID 对齐 prompt；
- sequence/title → SID prompt。

因此 `max_samples=5000` 表示先抽 5000 条原始训练记录，不代表最终 Trainer 里只有 5000 个 prompt。

## 建议怎么跑

第一次不建议直接上 20k。可以分三档：

```text
5k  : 验证训练是否稳定、reward 是否正常、显存和速度是否可接受
10k : 看 HR/NDCG 是否有稳定方向
20k : 在前两档确认有效后再做正式扩展
```

优先保持 `num_generations=8`，先缩小 prompt 数据量。因为 GRPO 的关键信号来自组内相对 reward，如果数据和 generation 数一起大幅缩减，很难判断是 RL 本身无效还是采样信号太弱。

## 评估要求

RL 跑完后不要换一套 Evaluation 协议。至少要和当前 SFT 结果保持：

```text
temporal Test
sample seed = 2024
sample size = 10000
constrained beam search
num_beams = 10
HR@1/3/5/10
NDCG@1/3/5/10
```

这样才能直接做 SFT 与 RL 的同协议对比。如果后续有算力，再把 SFT 和 RL 都补一次 full-Test Evaluation；不能只给 RL 换成全量评估然后和 10k SFT 结果直接比较。
