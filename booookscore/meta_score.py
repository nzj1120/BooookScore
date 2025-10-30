import argparse
from typing import List

from booookscore.score import Scorer

META_LABELS: List[str] = [
    "writing process",
    "formatting/meta",
    "plot speculation",
    "other meta",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summ_path", type=str, required=True, help="path to the summaries to audit")
    parser.add_argument("--annot_path", type=str, required=True, help="path to save meta commentary annotations")
    parser.add_argument("--api", type=str, choices=["openai", "anthropic", "together"], help="api to use")
    parser.add_argument("--api_key", type=str, help="API key string or path to a txt file storing it")
    parser.add_argument("--base_url", type=str, default=None, help="optional base url for OpenAI-compatible endpoints")
    parser.add_argument("--model", type=str, default="gpt-4", help="evaluator model")
    parser.add_argument("--v2", action="store_true", help="use batch mode (experimental)")
    parser.add_argument("--batch_size", type=int, help="batch size if v2 is used")
    parser.add_argument(
        "--template_path",
        type=str,
        default=None,
        help="prompt template for detecting meta commentary",
    )
    args = parser.parse_args()

    if args.template_path is None:
        if args.v2:
            template_path = "prompts/get_meta_annotations_v2.txt"
        else:
            template_path = "prompts/get_meta_annotations.txt"
    else:
        template_path = args.template_path

    scorer = Scorer(
        model=args.model,
        api=args.api,
        api_key=args.api_key,
        base_url=args.base_url,
        summ_path=args.summ_path,
        annot_path=args.annot_path,
        template_path=template_path,
        v2=args.v2,
        batch_size=args.batch_size,
        labels=META_LABELS,
        no_issue_token="no meta issues",
    )
    clean_score = scorer.get_score()
    meta_ratio = 1 - clean_score
    print(f"MetaContentScore = {clean_score}")
    print(f"MetaContentRate = {meta_ratio}")


if __name__ == "__main__":
    main()
