# MiniOneRec on MovieLens-1M

这个仓库把 [MiniOneRec](https://arxiv.org/abs/2510.24431) 的生成式推荐链路迁到了 MovieLens-1M，并完成了一次真实的 SFT 训练与离线评估。

当前进度很明确：**SFT 已跑完，RL 没跑。** RL 代码和实验方案保留在仓库里，后面有算力时可以从 SFT checkpoint 继续，不需要重训 SFT。

## 先看结果

本次实验使用 MovieLens-1M，按时间做 80/10/10 切分；SFT 训练完整跑了 2 个 epoch。为了控制评估成本，没有对约 9.9 万条 Test 全量做 50-beam 解码，而是在已经划分好的 Test 中用 `seed=2024` 固定抽取 10,000 条，使用 constrained beam search，`num_beams=10`。

| 指标 | 结果 |
| --- | ---: |
| HR@1 | 0.0121 |
| HR@3 | 0.0356 |
| HR@5 | 0.0603 |
| **HR@10** | **0.1135** |
| NDCG@1 | 0.0121 |
| NDCG@3 | 0.0256 |
| NDCG@5 | 0.0357 |
| **NDCG@10** | **0.0528** |
| invalid item count | **0** |

这里没有训练 SASRec、GRU4Rec 之类的 learned baseline，所以我不把这组数写成“优于某模型”。为了确认模型不是随机生成，给一个最简单的随机参照：过滤后 Item 数是 3416，如果从全 Item 集均匀随机取 K 个不同物品，单目标任务的期望 `HR@K = K / 3416`。

| K | 本项目 HR@K | 均匀随机 HR@K | 本项目 / 随机期望 |
| ---: | ---: | ---: | ---: |
| 1 | 1.21% | 0.0293% | 41.3× |
| 3 | 3.56% | 0.0878% | 40.5× |
| 5 | 6.03% | 0.1464% | 41.2× |
| 10 | **11.35%** | **0.2927%** | **38.8×** |

这个随机值只是 sanity check，不是正式模型 baseline。它能说明 SFT 学到了明显高于随机的用户序列信号，但不能说明模型已经超过成熟的序列推荐方法。

完整实验记录见 [`experiments/20260910_sft/`](experiments/20260910_sft/README.md)。

## 这个项目具体做了什么

这个仓库不是简单把原项目的数据集名字换成 MovieLens。相对 upstream，主要工作集中在下面几块。

**1. 把 MovieLens-1M 接进生成式推荐链路。** 直接读取 GroupLens 的 `ratings.dat / movies.dat / users.dat`，对 user/item 做迭代 5-core，按时间构造 next-item 样本，再做全局时间 80/10/10 切分。模型输入只使用交互历史和电影的标题、类别信息；`users.dat` 里的性别、年龄、职业等人口属性没有喂给模型。

**2. 重新构造 Item Semantic ID。** 电影标题与 genres 先经过 Qwen3-Embedding-0.6B 得到 1024 维语义向量，再用三层 RQ-VAE 压成 `<a_x><b_y><c_z>`。实际训练中依次试了 `[256,256,256]`、`[512,512,512]`、`[1024,1024,1024]` 三组码本，SID collision 从 13 降到 6，再降到 3。最后 3416 个 Item 得到 3413 个唯一 SID，collision rate 为 0.000878。

**3. 跑通完整 SFT，而不是只做数据和脚本适配。** Base model 使用 Qwen2.5-1.5B，历史长度最多 10，训练样本 794,856 条，2 epoch。在 A800 80GB 上把 micro batch 从 4 调到 8，同时把 gradient accumulation 从 32 调到 16，effective batch 仍然保持 128。服务器日志中单步训练速度从前一版约 3.6 s/it 降到后续约 2.1–2.4 s/it。最终 SFT runtime 约 16 小时 54 分钟，Trainer 报告 train loss 0.6962。

**4. 把生成结果限制在合法 Item 空间。** Evaluation 不是让 LLM 自由输出字符串，而是根据全部合法 SID 构造 prefix tree，在解码阶段做约束。最终 10k 评估里 `invalid_item_count=0`。注意代码里的 `CC` 表示非法预测计数，不是前面 RQ-VAE 的 SID collision，两者不要混在一起。

**5. 把长时间实验做成可恢复的工程链路。** `run.sh` / `scripts/pipeline.py` 负责数据、Embedding、RQ-VAE、SID、SFT、Evaluation 和可选 RL；每个阶段有真实输出校验和恢复逻辑，SFT/RL 可以从最新 checkpoint 继续。数据、模型、cache、checkpoint 默认放在独立数据盘，不塞进 Git 仓库。

## 数据是怎么变成训练样本的

原始 MovieLens-1M 大约有 100 万条评分记录。本项目把评分记录当作用户与电影的交互，不把评分值本身作为排序特征。处理后数据统计如下：

```text
6040 users
3416 items
999611 interactions after iterative 5-core

794856 train samples
 99357 valid samples
 99358 test samples
```

对每个用户先按时间排序，然后用“历史 → 下一物品”构造样本。例如：

```text
[i1]             -> i2
[i1, i2]         -> i3
[i1, i2, i3]     -> i4
...
```

历史最多保留最近 10 个 Item。构造完成后，再按照 target timestamp 对所有样本做全局时间切分，所以 Validation/Test 的 target 都发生在 Train 之后。

## 模型链路

```text
MovieLens-1M
    │
    ├─ iterative 5-core + chronological next-item samples
    │
    ├─ title + genres
    │      ↓
    │  Qwen3-Embedding-0.6B
    │      ↓ 1024-d
    │  RQ-VAE (3 codebooks)
    │      ↓
    │  Semantic ID: <a_x><b_y><c_z>
    │
user recent history (max 10)
    │
    ↓
Qwen2.5-1.5B SFT
    │
    ↓
constrained SID generation
    │
    ↓
Top-K recommended items
```

它和传统“先有候选集，再给候选打分”的排序模型不太一样。这里的 SFT 模型直接根据用户历史生成下一 Item 的 Semantic ID；beam search 输出多个合法 SID 后，自然形成 Top-K 推荐结果。因此这个项目更接近**生成式召回 / 生成式推荐**，而不是传统 CTR 精排。

## 如何复现

```bash
git clone https://github.com/Nekotire/-MovieLens1M-MiniOneRec-.git MovieLens1M-MiniOneRec
cd MovieLens1M-MiniOneRec
bash setup.sh
bash run.sh
```

默认配置在 [`config/movielens1m.yaml`](config/movielens1m.yaml)。当前默认配置对应这次已经验证过的 SFT 路径：RQ-VAE 使用 1024×3 码本、SFT micro batch 8、RL 默认关闭、Evaluation 固定抽 10k Test 并使用 beam=10。若要跑全量 Test，可把 `evaluation.sample_size` 设为 `0`，再自行提高 beam 数。

中断后再次运行 `bash run.sh` 会检查已有输出并从可恢复的位置继续。只想从某个阶段开始可以使用：

```bash
bash run.sh --from-stage 8
```

## RL：保留设计，但本次没有执行

RL 不是本次结果的一部分，仓库里也没有任何 RL 指标。当前代码的思路是：从 SFT `final_checkpoint` 出发，用 GRPO 做 recommendation-oriented post-training。`reward_type=ranking` 时同时使用 exact-hit reward 和组内排名/NDCG 风格 reward；每个 prompt 生成多个候选，在组内比较 reward，再结合 reference model / KL 约束更新 policy。

RL 的数据构造、reward、GRPO 参数、建议的 5k→20k 分阶段实验方式都单独写在 [`docs/RL_EXTENSION.md`](docs/RL_EXTENSION.md)。
## 仓库怎么读

核心链路只需要先看这些文件：

```text
config/movielens1m.yaml              # 主配置
data/movielens1m_data_process.py     # MovieLens 预处理
rq/text2emb/item_text2emb.py         # Item 文本向量
rq/rqvae.py                          # RQ-VAE 训练
rq/generate_indices.py               # Semantic ID 生成
convert_dataset.py                   # 转 MiniOneRec 数据格式
sft.py                               # SFT
minionerec_utils/constrained_decoding.py
evaluate.py                          # constrained beam search
scripts/evaluate_metrics.py          # HR/NDCG
scripts/pipeline.py                  # 端到端编排
```

仓库还保留了 upstream 的 GPR、TS-Rec、SASRec、RQ-Kmeans 等研究分支，默认 pipeline 不会调用。详细文件说明见 [`docs/CODE_MAP.md`](docs/CODE_MAP.md)。

## 已知限制

这次实验有三个边界需要提前说明。第一，没有 learned baseline，所以随机推荐只能用来确认模型明显不是随机，不能替代 SASRec 等正式对照。第二，最终 RQ-VAE 仍有 3 个 SID collision；为了控制实验时间，本次允许最多 3 个 collision 后继续 SFT，并在结果中原样保留这个事实。第三，报告的 HR/NDCG 来自 temporal Test 中固定抽取的 10,000 条，不是 99,358 条 Test 的全量评估。

## Acknowledgements

本项目基于 MiniOneRec 开源代码改造，并保留原项目 Apache-2.0 License。MiniOneRec 原论文：

```bibtex
@misc{MiniOneRec,
  title={MiniOneRec: An Open-Source Framework for Scaling Generative Recommendation},
  author={Xiaoyu Kong and Leheng Sheng and Junfei Tan and Yuxin Chen and Jiancan Wu and An Zhang and Xiang Wang and Xiangnan He},
  year={2025},
  eprint={2510.24431},
  archivePrefix={arXiv},
  primaryClass={cs.IR}
}
```
