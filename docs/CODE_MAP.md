# 代码地图

仓库保留了 MiniOneRec upstream 的多条研究分支。为了不把主链路和可选实验混在一起，这里按用途列出来。

## 主链路

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 正式 pipeline 入口 |
| `setup.sh` | 环境和数据盘准备 |
| `config/movielens1m.yaml` | MovieLens-1M 主配置 |
| `scripts/pipeline.py` | 13 个阶段的编排、恢复、校验 |
| `data/movielens1m_data_process.py` | 5-core、时序样本、80/10/10 split |
| `rq/text2emb/item_text2emb.py` | title + genres → Item embedding |
| `rq/rqvae.py` | RQ-VAE 训练 |
| `rq/generate_indices.py` | Item embedding → Semantic ID |
| `convert_dataset.py` | 转换到 MiniOneRec SFT 格式 |
| `sft.py` | Qwen2.5-1.5B SFT |
| `evaluate.py` | constrained beam search |
| `scripts/evaluate_metrics.py` | HR/NDCG/invalid item 统计 |
| `scripts/sample_evaluation_csv.py` | 固定 seed 的 sampled Test |

## 共用工具

`minionerec_utils/` 放 pipeline 和训练共同使用的轻量工具：配置解析、checkpoint、constrained decoding、metrics、RL sampling 等。

## RL 扩展

- `rl.py`：GRPO 主训练入口；
- `minionerec_trainer.py`：自定义 Trainer；
- `minionerec_utils/rl_data.py`：训练数据固定 seed reservoir sampling；
- `docs/RL_EXTENSION.md`：本仓库后续 RL 计划。

当前默认配置 `rl.enabled=false`，所以不会误触发高成本 RL。

## 保留的 upstream / 可选研究分支

下面这些代码保留用于进一步研究，但不在默认 MovieLens pipeline 中：

- `rq/rqkmeans_*.py`：RQ-Kmeans 及 constrained/plus 版本；
- `sft_gpr.py` / `rl_gpr.py`：GPR 路径；
- `ts_rec_*.py`：TS-Rec 路径；
- `sasrec.py` / `SASRecModules_ori.py`：SASRec 相关实现；
- `calc.py`、`merge.py`、`split.py` 等 upstream 辅助脚本。

暂时不大规模移动这些文件，主要是避免破坏 upstream import 和后续对照。主链路入口已经集中在上面的“主链路”部分。
