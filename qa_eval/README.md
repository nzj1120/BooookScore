# QA Eval

`qa_eval` 提供一个与 `align_eval` 平行的工具，用于通过跨文本问答的方式衡量书籍摘要的忠实度。工具会让大模型分别从原文与摘要中各生成若干问答对，然后互相检验答案是否能在另一端找到支撑，从而给出两个分数：

- **Hallucination Score**：摘要侧的问题去询问原文，衡量摘要是否凭空添加了原文没有的信息。
- **Coverage Score**：原文侧的问题去询问摘要，衡量摘要是否覆盖了原文的重要情节。

## 安装依赖

与 BooookScore 其余组件相同，确保安装了可用的大模型 API 依赖，例如：

```bash
pip install openai anthropic tqdm
```

## 输入格式

与 `align_eval`、`booookscore.score` 相同，输入均为 JSON：顶层是一个对象，键为书名（通常是 EPUB 文件名），值为对应的完整文本/摘要字符串。

## 命令行示例

```bash
python -m qa_eval.cli \
  --source_path data/originals.json \
  --summary_path summaries/my-summary.json \
  --output_path reports/qa-eval.json \
  --api openai \
  --api_key empty \
  --base_url http://10.244.2.114:8808/v1 \
  --model reportify \
  --question_count 5 \
  --show_progress
```

关键参数说明：

- `--question_count`：每边生成的问题数量，默认 5。
- `--max_retries`：当模型输出不符合 JSON 规范时的重试次数，默认 3（始终至少尝试一次）。
- `--generation_model` / `--judge_model`：如需使用不同模型生成问题和做判定，可分别覆盖。
- `--generation_max_tokens` / `--judge_max_tokens`：控制生成与判定阶段的最大输出长度。

## 输出结果

命令会在终端打印宏观指标，并将完整报告写入 `--output_path`：

- `macro_metrics.hallucination_score`：摘要问题在原文中被判定为正确的比例，越高表示越少幻觉。
- `macro_metrics.coverage_score`：原文问题在摘要中被判定为正确的比例，越高表示覆盖度更好。
- `books[...]`：逐书的详细问答与判定记录，包括模型裁决、理由和对方文本中给出的答案。

## Python API

```python
from qa_eval.api import ModelSpec
from qa_eval.eval import EvaluationConfig, GenerationConfig, JudgeConfig, evaluate_book

spec = ModelSpec(api="openai", api_key="empty", model="reportify", base_url="http://10.244.2.114:8808/v1")
config = EvaluationConfig(
    question_count=5,
    generation=GenerationConfig(model=spec, temperature=0.2, max_tokens=2048),
    judge=JudgeConfig(model=spec, temperature=0.0, max_tokens=512),
)
report = evaluate_book("my-book", source_text, summary_text, config)
print(report.hallucination_score, report.coverage_score)
```

## 与其它组件的关系

`qa_eval` 独立于 BooookScore 与 AlignEval，可单独运行；它复用了相同的 JSON 输入结构，方便与现有数据流水线互通。
