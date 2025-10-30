# AlignEval

AlignEval 是一个用于评估“书籍缩写是否忠于原著”的命令行工具与 Python 库。它基于中文 BERT 句向量和单调对齐，输出四项指标：

- **Coverage**：自动阈值覆盖率
- **Alignment Confidence**：对齐句对的平均相似度
- **PFS (Position Fidelity Score)**：位置信度幂次映射
- **SCS (Stitching Compactness Score)**：句级拼接紧凑度（Top-K=3）

## 安装依赖

项目根目录已包含所需依赖（`transformers`, `torch`, `numpy`, `tqdm` 等）。若在独立环境中使用，请确保已安装：

```bash
pip install torch transformers numpy tqdm
```

## 输入数据格式

与 BooookScore `score` 命令相同，AlignEval 读取两个 JSON：

- `source_path`：原文（书名 → 完整文本）
- `summary_path`：缩写/摘要（书名 → 完整摘要）

两个 JSON 应具有相同的键集合，每条记录会按句对齐并计算指标。

## 快速开始

```bash
python -m align_eval.cli \
  --source_path data/original.json \
  --summary_path summaries/my_summary.json \
  --output_path reports/result.json \
  --model_name hfl/chinese-bert-wwm-ext \
  --batch_size 8 \
  --alignment nw \
  --bandwidth 4 \
  --pfs_gamma 3.0 \
  --alpha 10 \
  --scs_beta 0.1
```

命令会输出指标摘要，并将详细 JSON 报告写入 `--output_path`。

## Python API

```python
from align_eval import EvaluationConfig, evaluate_alignment, evaluate_corpus

config = EvaluationConfig()
reports, macro = evaluate_corpus(source_map, summary_map, config)
```

更多示例可参见 `align_eval/eval.py`。

