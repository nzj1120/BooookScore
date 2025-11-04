# AlignEval

AlignEval 是一个用于评估“书籍缩写是否忠于原著”的命令行工具与 Python 库。它基于中文 BERT 句向量和单调对齐，输出四项指标：

- **Coverage**：自动阈值覆盖率
- **Alignment Confidence**：对齐句对的平均相似度
- **PFS (Position Fidelity Score)**：位置信度幂次映射
- **SCS (Stitching Compactness Score)**：句级拼接紧凑度（Top-K=3）

## 指标公式

假设摘要共有 \(M\) 句 \(s_i\)，原文共有 \(N\) 句 \(x_j\)。句向量相似度矩阵记为 \(\mathrm{sim}(s_i, x_j)\)。

### Coverage（覆盖率）

\[
t_i = \max_j \mathrm{sim}(s_i, x_j), \qquad \mathrm{TH} = \frac{1}{M} \sum_{i=1}^{M} t_i
\]

Coverage 使用动态阈值 \(\mathrm{TH}\)：

\[
\mathrm{Coverage} = \frac{1}{M} \sum_{i=1}^{M} \mathbf{1}[t_i \ge \mathrm{TH}]
\]

### Alignment Confidence（对齐置信度）

对齐算法返回的路径为句对集合 \(\{(i, j_i)\}\)。对齐置信度是这些句对的平均相似度：

\[
\mathrm{Alignment\ Confidence} = \frac{1}{|\mathcal{P}|} \sum_{(i, j_i) \in \mathcal{P}} \mathrm{sim}(s_i, x_{j_i})
\]

### PFS（Position Fidelity Score）

对齐句对 \((i, j_i)\) 的归一化位置为

\[
u_i = \frac{i + 0.5}{M}, \qquad v_i = \frac{j_i + 0.5}{N}, \qquad d_i = |u_i - v_i|
\]

权重取相似度下界 \(\epsilon\)：

\[
w_i = \max(\mathrm{sim}(s_i, x_{j_i}), \epsilon), \qquad \bar{D} = \frac{\sum_i w_i d_i}{\sum_i w_i}
\]

幂次映射得到

\[
\mathrm{PFS} = (1 - \bar{D})^{\gamma}
\]

其中 \(\gamma\) 由 `--pfs_gamma` 控制（默认 3.0）。

### SCS（Stitching Compactness Score，Top-K=3）

对每个摘要句选择原文中相似度最高的三句，记索引为 \(j_{ik}\)，相似度为 \(s_{ik}\)。归一化位置：

\[
x_{ik} = \frac{j_{ik} + 0.5}{N}
\]

Softmax 权重：

\[
p_{ik} = \frac{\exp(\alpha s_{ik})}{\sum_{r=1}^{3} \exp(\alpha s_{ir})}
\]

加权均值与方差：

\[
\mu_i = \sum_{k=1}^{3} p_{ik} x_{ik}, \qquad \sigma_i^2 = \sum_{k=1}^{3} p_{ik} (x_{ik} - \mu_i)^2
\]

风险值与逐句得分：

\[
r_i = \mathrm{clip}\!\left(\frac{\sigma_i^2}{\beta}, 0, 1\right), \qquad \mathrm{SCS}_i = 1 - r_i
\]

全局 SCS 是所有 \(\mathrm{SCS}_i\) 的均值。参数 \(\alpha\)、\(\beta\) 分别对应 `--alpha`、`--scs_beta`。

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

命令会输出指标摘要，并将详细 JSON 报告写入 `--output_path`。执行过程中会显示“Evaluating summaries”进度条，方便在大批量书目上查看处理进度。

## 使用本地已下载的中文 BERT

如果你已经在本地下载好了中文 BERT 权重（例如通过 `git clone https://huggingface.co/hfl/chinese-bert-wwm-ext`），只需在运行时把模型目录传给 `--model_name` 即可：

```bash
# 假设权重目录就在当前文件夹，例如 ./chinese-bert-wwm-ext
HF_HUB_OFFLINE=1 \
python -m align_eval.cli \
  --source_path data/original.json \
  --summary_path summaries/my_summary.json \
  --output_path reports/result.json \
  --model_name ./chinese-bert-wwm-ext
```

目录需要包含 `config.json`、`pytorch_model.bin`、`vocab.txt` 等文件，`transformers` 会直接从该路径加载，不会再访问网络。若希望集中存放权重，也可以设置 `TRANSFORMERS_CACHE` 环境变量或在配置里传入其它本地模型目录。`HF_HUB_OFFLINE=1` 是可选项，用于强制禁用网络请求，避免因为代理或镜像证书问题导致加载失败。

## Python API

```python
from align_eval import EvaluationConfig, evaluate_alignment, evaluate_corpus

config = EvaluationConfig()
reports, macro = evaluate_corpus(source_map, summary_map, config, show_progress=True)
```

将 `show_progress` 设为 `True` 可以在批量评估时获得与 CLI 相同的进度条。更多示例可参见 `align_eval/eval.py`。

