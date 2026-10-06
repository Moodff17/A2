"""
Main.py - single entry point for the whole project.

    python Main.py process            build Dataset/Processed/processed_urls.csv from the raw data
    python Main.py explore            exploratory analysis + figures
    python Main.py baseline           sanity-check baselines
    python Main.py train              train + compare classifiers, save the best model
    python Main.py cluster [--k 5]    cluster the phishing URLs (no labels used)
    python Main.py evaluate           robustness checks and error analysis
    python Main.py predict URL [URL]  score one or more URLs (offline)
    python Main.py all                process -> explore -> baseline -> train -> cluster -> evaluate
"""
import argparse


def run(command, extra):
    # imports are inside so `predict` stays fast and does not need the training libraries
    if command == "process":
        from src import Prepare_Data; Prepare_Data.main()
    elif command == "explore":
        from src import Explore_Features; Explore_Features.main()
    elif command == "baseline":
        from src import Sanity_Baseline; Sanity_Baseline.main()
    elif command == "train":
        from src import Train_Classifiers; Train_Classifiers.main()
    elif command == "cluster":
        from src import Cluster_Phishing; Cluster_Phishing.main(extra.k)
    elif command == "evaluate":
        from src import Evaluate_Robustness; Evaluate_Robustness.main()
    elif command == "predict":
        from src import Predict; Predict.main(extra.urls)
    elif command == "all":
        for step in ("process", "explore", "baseline", "train", "cluster", "evaluate"):
            print(f"\n{'=' * 20} {step.upper()} {'=' * 20}")
            run(step, extra)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing URL Detection (COS30049 Assignment 2)")
    parser.add_argument("command", choices=["process", "explore", "baseline", "train", "cluster",
                                            "evaluate", "predict", "all"])
    parser.add_argument("urls", nargs="*", help="URLs to score (predict only)")
    parser.add_argument("--k", type=int, default=None, help="number of clusters (cluster only)")
    args = parser.parse_args()
    run(args.command, args)
