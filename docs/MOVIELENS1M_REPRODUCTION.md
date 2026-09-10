# MovieLens-1M reproduction notes

这份文档只讲怎么复现和排错；实验结果放在 `experiments/20260910_sft/`。

## Default workflow

```bash
git clone https://github.com/Nekotire/-MovieLens1M-MiniOneRec-.git MovieLens1M-MiniOneRec
cd MovieLens1M-MiniOneRec
bash setup.sh
bash run.sh
```

默认 pipeline 共 13 个阶段：Environment → Download → Preprocess → Item Embedding → RQ-VAE → Semantic ID → Dataset Conversion → SFT → SFT Evaluation → RL → RL Evaluation → Comparison → Report。

当前 `config/movielens1m.yaml` 里 `rl.enabled=false`，所以默认实验会完成到 SFT Evaluation，RL 阶段明确跳过，最后报告会显示 `RL: NOT EXECUTED`。要做 RL 再打开开关即可。

## Data root

生成数据和模型不放 Git 工作区。`setup.sh` 会优先寻找可写的大容量挂载盘，也可以手动：

```bash
export DATA_ROOT=/root/autodl-tmp
```

典型目录：

```text
$DATA_ROOT/
├── downloads/
├── datasets/MovieLens1M/
│   ├── raw/ml-1m/
│   ├── processed/
│   └── minionerec/
├── output/MovieLens1M/
├── results/MovieLens1M/
└── models/ or huggingface cache
```

## Resume

pipeline 不只看 `.state/*.done`。marker 存在时还会检查真实输出；文件缺失或不合法就会重跑该阶段。SFT/RL 会寻找数值最大的 `checkpoint-*` 继续训练。

```bash
bash run.sh                 # normal resume
bash run.sh --from-stage 8  # start checking/running from SFT
bash run.sh --force         # ignore stage markers, keep trainer checkpoints
bash run.sh --fresh         # clear generated model/results/state under DATA_ROOT
```

## MovieLens preprocessing

- official GroupLens MovieLens-1M files；
- user/item 独立 iterative k-core，默认都是 5；
- user 内部按 timestamp 稳定排序；
- history 最多最近 10 个 Item；
- target sample 再按 target timestamp 做全局时间切分；
- 默认 train/valid/test = 80/10/10；
- 用户人口属性不作为模型输入。

当前真实处理统计：6040 users、3416 items、999611 interactions；样本数 794856 / 99357 / 99358。

## SID gate

`rqvae.max_collisions` 控制 SFT 前允许的最大 SID collision 数。严格实验可以设为 0；本次实际 MovieLens run 在 `[1024,1024,1024]` 码本下得到 3 个 collision，因此默认复现实验使用 3，并在 README/experiment record 中明确披露。

## Cost-aware evaluation

默认配置：

```yaml
evaluation:
  sample_size: 10000
  sample_seed: 2024
  cutoffs: [1, 3, 5, 10]
  batch_size: 4
  num_beams: 10
```

`sample_size > 0` 时，pipeline 会先从 temporal Test 固定抽样，再做 constrained beam search。`sample_size: 0` 表示 full-test evaluation。

由于同一条 target 只有一个 relevant Item，HR@K 就表示真实下一 Item 是否进入 Top-K；NDCG@K 还会根据命中的位置折扣。`invalid_item_count` / `CC` 检查生成 SID 是否属于合法 Item 集。

## Common issues

- GroupLens HTTPS 下载失败：先确认服务器能访问官方源；不要静默切第三方镜像。
- SFT 前被 SID gate 卡住：查看 `MovieLens1M.index.stats.json`，不要把 collision 当成 tokenizer/CC 问题。
- Evaluation 很慢：生成式 beam search 比 teacher-forced validation 贵很多，可以先用固定 10k Test 子集。
- 服务器断开：先确认 `final_checkpoint` 是否存在，再决定从哪个 stage 恢复。
