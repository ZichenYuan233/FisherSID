from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("--output", default="summary.json")
    args = parser.parse_args()
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["method"], []).append(row)
    summary = {}
    for method, values in grouped.items():
        summary[method] = {}
        for metric in [
            "ranking_kl", "agree_at_10", "ndcg_at_10", "recall_at_20",
            "collision_rate", "codebook_utilization",
        ]:
            samples = [row[metric] for row in values]
            summary[method][metric] = {
                "mean": mean(samples), "sample_sd": stdev(samples) if len(samples) > 1 else 0.0,
            }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output = Path(args.output)
    metrics_order = ["ranking_kl", "agree_at_10", "ndcg_at_10", "recall_at_20", "collision_rate", "codebook_utilization"]
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", *metrics_order])
        for method, metrics in summary.items():
            writer.writerow([method, *[f"{metrics[key]['mean']:.4f} +/- {metrics[key]['sample_sd']:.4f}" for key in metrics_order]])
    latex = [
        "\\begin{tabular}{lrrrrrr}", "\\toprule",
        "Method & KL$\\downarrow$ & Agree@10$\\uparrow$ & N@10$\\uparrow$ & R@20$\\uparrow$ & Coll.$\\downarrow$ & Util.$\\uparrow$ \\\\",
        "\\midrule",
    ]
    for method, metrics in summary.items():
        cells = [f"{metrics[key]['mean']:.4f} $\\pm$ {metrics[key]['sample_sd']:.4f}" for key in metrics_order]
        latex.append(method.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    output.with_suffix(".tex").write_text("\n".join(latex), encoding="utf-8")
    for method, metrics in summary.items():
        print(method, " ".join(f"{key}={value['mean']:.4f}+/-{value['sample_sd']:.4f}" for key, value in metrics.items()))


if __name__ == "__main__":
    main()
