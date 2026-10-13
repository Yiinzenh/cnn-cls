"""实验入口：数据集统计、基础模型训练、对照实验与最终评估。"""
import argparse
import json
from pathlib import Path

import pandas as pd

from config.setting import (BEST, CLASSES, DEVICE, LOG_DIR, SEED, build_args,
                            set_seed)
from data.dataset import prepare_loaders, save_summary, scan, summarize
from engine.evaluator import evaluate
from engine.trainer import Trainer, load_model
from model.cnn import build_model, count_params, describe
from utils.plot import (plot_class_distribution, plot_confusion, plot_curves,
                        plot_misclassified, plot_sample_grid)


def task_stats():
    """任务1：统计数据集信息，观察样本，确认输入维度。"""
    records = scan()
    info = summarize(records)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    save_summary(info)

    plot_class_distribution(info["class_counts"],
                            "Class distribution of RetinalOCT dataset",
                            "class_distribution.png")

    picked = [next(r for r in records["train"] if r["name"] == c) for c in CLASSES]
    plot_sample_grid([r["path"] for r in picked], [r["name"] for r in picked],
                     "Sample images of RetinalOCT dataset", "samples.png",
                     cols=4, rows=2)

    for split in info["split_counts"]:
        print(f"{split}: total={info['split_counts'][split]}  "
              f"{info['class_counts'][split]}")
    return info


def task_train(tag="baseline"):
    """任务3：训练最优模型，记录曲线并保存权重与日志。"""
    set_seed(SEED)
    args = build_args(**BEST)
    records, loaders = prepare_loaders(args["batch_size"])
    model = build_model(args)
    for line in describe(args, model):
        print(line)
    print(f"可训练参数量: {count_params(model)}  设备: {DEVICE}")

    trainer = Trainer(model, args, device=DEVICE, tag=tag)
    trainer.fit(loaders["train"], loaders["val"], args["epochs"])
    trainer.save(extra={"params": count_params(model), "group": "best", "value": "base"})
    plot_curves(trainer.history, f"{tag}  best val acc {trainer.best_val_acc:.4f}",
                f"curves_{tag}.png")
    print(f"最优验证集精度 {trainer.best_val_acc:.4f} (epoch {trainer.best_epoch})")
    return trainer


def task_search(groups=None):
    """任务4：执行控制变量法对照实验。"""
    from experiments.search import GROUPS, best, search
    selected = {k: v for k, v in GROUPS.items() if groups is None or k in groups}
    table = search(selected)
    print(f"验证集最优模型: {best(table)}")
    return table


def task_eval(tag):
    """任务5：用最优模型在测试集上评估，输出混淆矩阵与错分样本。"""
    record = json.loads((Path(LOG_DIR) / f"{tag}.json").read_text(encoding="utf-8"))
    args = record["args"]
    records, loaders = prepare_loaders(args["batch_size"])

    model = build_model(args)
    load_model(model, tag)
    result = evaluate(model, loaders["test"], records=records["test"])

    print(f"[{tag}] 测试集准确率: {result['accuracy']:.4f}")
    table = pd.DataFrame(result["per_class"])
    print(table.to_string(index=False))
    table.to_csv(Path(LOG_DIR) / f"eval_{tag}.csv", index=False, encoding="utf-8-sig")

    plot_confusion(result["confusion_matrix"], f"Confusion matrix ({tag})",
                   f"confusion_{tag}.png")
    mis = pd.DataFrame(result["misclassified"])
    mis.to_csv(Path(LOG_DIR) / f"misclassified_{tag}.csv", index=False,
               encoding="utf-8-sig")
    if not mis.empty:
        plot_misclassified(result["misclassified"], f"Misclassified samples ({tag})",
                           f"misclassified_{tag}.png")

    (Path(LOG_DIR) / f"eval_{tag}.json").write_text(
        json.dumps({"tag": tag, "accuracy": result["accuracy"],
                    "per_class": result["per_class"],
                    "confusion_matrix": result["confusion_matrix"].tolist(),
                    "misclassified_count": len(result["misclassified"])},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="OCTDL 自定义卷积神经网络图像分类实验")
    sub = parser.add_subparsers(dest="task", required=True)

    sub.add_parser("stats", help="任务1 数据集统计与预处理")

    p_train = sub.add_parser("train", help="任务3 训练基础模型")
    p_train.add_argument("--tag", default="baseline")

    p_search = sub.add_parser("search", help="任务4 控制变量对照实验")
    p_search.add_argument("--groups", nargs="*", default=None,
                          help="仅运行指定实验组，如 kernel depth dropout lr pool epoch")

    p_eval = sub.add_parser("eval", help="任务5 测试集评估")
    p_eval.add_argument("--tag", required=True)

    args = parser.parse_args()
    if args.task == "stats":
        task_stats()
    elif args.task == "train":
        task_train(args.tag)
    elif args.task == "search":
        task_search(args.groups)
    elif args.task == "eval":
        task_eval(args.tag)


if __name__ == "__main__":
    main()