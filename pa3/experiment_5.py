import csv
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from src.ann.activations.linear import Linear
from src.ann.clf import ANNC
from src.ann.data.c45 import load_spam, load_volcanoes, load_voting
from src.ann.data.header import FeatureType
from src.ann.optimizers.sgd import SGDOptimizer



OUTPUT_DIR = os.path.join("output", "experiment_5")


def make_logger(section: str) -> logging.Logger:
    logger = logging.getLogger(f"experiment_5{section}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, f"experiment_5{section}.txt"), mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def sep(logger: logging.Logger):
    logger.info("-"*100)


def experiment_a():
    section = "a"
    logger = make_logger(section)
    logger.info("Experiment (a):\n")
    logger.info("Model: 1 hidden unit, Linear hidden activation (perceptron approximation)")
    logger.info("Output: Sigmoid + binary cross-entropy")
    logger.info("gamma: 0")
    logger.info("lr: 0.01")
    logger.info("epsilon: 1e-6")
    logger.info("max_iters: 1000000")
    logger.info("Folds: 5")
    logger.info("Seed: 12345")
    logger.info("Preprocessing: StandardScaler on continuous features only")
    logger.info("Init: np.random.seed(12345) before every model")

    loaders = {
        "voting": load_voting,
        "spam": load_spam,
        "volcanoes": load_volcanoes,
    }
    metric_names = ["aroc", "precision", "recall", "accuracy"]
    lr = 1e-2
    epsilon = 1e-5
    max_iters = 100_000
    gamma = 0.0

    results = {
        dataset_name: {metric_name: [] for metric_name in metric_names}
        for dataset_name in loaders
    }
    oof = {dataset_name: {"y": [], "score": []} for dataset_name in loaders}

    for dataset_name, loader in loaders.items():
        sep(logger)
        logger.info(f"{dataset_name.capitalize()} dataset")
        header, X, y_gt = loader()
        X = np.asarray(X, dtype=float)
        y_gt = np.asarray(y_gt).reshape(-1)

        n_cont = 0
        n_disc = 0
        n_inputs = 0
        cont_idx = []
        for feature_idx, feature in enumerate(header.features):
            if feature.type == FeatureType.DISCRETE:
                n_disc += 1
                n_inputs += len(feature.domain())
            else:
                n_cont += 1
                n_inputs += 1
                cont_idx.append(feature_idx)

        classes, counts = np.unique(y_gt, return_counts=True)
        logger.info(f"Shape: {X.shape}")
        logger.info(f"Class counts: {dict(zip(classes.tolist(), counts.tolist()))}")
        logger.info(f"Continuous features: {n_cont}")
        logger.info(f"Discrete features: {n_disc}")
        logger.info(f"Network inputs: {n_inputs}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_gt), start=1):
            logger.info(f"Fold: {fold}")
            logger.info(f"Train size: {len(train_idx)}")
            logger.info(f"Test size: {len(test_idx)}")

            X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
            y_train, y_test = y_gt[train_idx], y_gt[test_idx]

            if cont_idx:
                scaler = StandardScaler()
                X_train[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
                X_test[:, cont_idx] = scaler.transform(X_test[:, cont_idx])

            np.random.seed(12345)
            model = ANNC(header, 1, Linear())
            optimizer = SGDOptimizer(model.parameters(), lr)

            t0 = time.perf_counter()
            model.fit(X_train, y_train, optimizer, gamma=gamma, epsilon=epsilon, max_iters=max_iters)
            elapsed = time.perf_counter() - t0

            train_accuracy = accuracy_score(y_train, model.predict(X_train))
            logger.info(f"Train time (s): {elapsed:.2f}")
            logger.info(f"Train accuracy: {train_accuracy}")

            y_score = model.predict_proba(X_test).reshape(-1)
            y_hat = model.predict(X_test)
            aroc = roc_auc_score(y_test, y_score)
            precision = precision_score(y_test, y_hat, zero_division=0)
            recall = recall_score(y_test, y_hat, zero_division=0)
            accuracy = accuracy_score(y_test, y_hat)

            results[dataset_name]["aroc"].append(aroc)
            results[dataset_name]["precision"].append(precision)
            results[dataset_name]["recall"].append(recall)
            results[dataset_name]["accuracy"].append(accuracy)
            oof[dataset_name]["y"].append(y_test)
            oof[dataset_name]["score"].append(y_score)

            logger.info(f"AROC: {aroc}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"Accuracy: {accuracy}")

        for metric_name in metric_names:
            values = np.asarray(results[dataset_name][metric_name])
            logger.info(f"Mean {metric_name}: {values.mean()}")
            logger.info(f"Variance {metric_name}: {values.var()}")

    sep(logger)
    logger.info("Summary (mean and variance across 5 folds):")
    col_width = {"aroc": 14, "precision": 16, "recall": 14, "accuracy": 16}
    header_row = f"{'dataset':<12}"
    for metric_name in metric_names:
        width = col_width[metric_name]
        header_row += f"{metric_name + ' mean':>{width}}{metric_name + ' var':>{width}}"
    logger.info(header_row)
    for dataset_name in loaders:
        row = f"{dataset_name:<12}"
        for metric_name in metric_names:
            values = np.asarray(results[dataset_name][metric_name])
            width = col_width[metric_name]
            row += f"{values.mean():>{width}.6f}{values.var():>{width}.6f}"
        logger.info(row)

    summary_path = os.path.join(OUTPUT_DIR, "experiment_5a.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["dataset"]
        for metric_name in metric_names:
            header.extend([f"{metric_name} mean", f"{metric_name} var"])
        writer.writerow(header)
        for dataset_name in loaders:
            row = [dataset_name]
            for metric_name in metric_names:
                values = np.asarray(results[dataset_name][metric_name])
                row.extend([f"{values.mean():.6f}", f"{values.var():.6f}"])
            writer.writerow(row)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(metric_names))
    width = 0.25
    for i, dataset_name in enumerate(loaders):
        means = [np.mean(results[dataset_name][metric_name]) for metric_name in metric_names]
        stds = [np.std(results[dataset_name][metric_name]) for metric_name in metric_names]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3, label=dataset_name)
    ax.set_xticks(x, metric_names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "experiment_5a.png"), dpi=150)
    plt.close(fig)

    plt.figure(figsize=(6, 5))
    for dataset_name in loaders:
        y_true = np.concatenate(oof[dataset_name]["y"])
        y_score = np.concatenate(oof[dataset_name]["score"])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        mean_aroc = np.mean(results[dataset_name]["aroc"])
        plt.plot(fpr, tpr, label=f"{dataset_name} (mean AROC={mean_aroc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "experiment_5a_roc.png"), dpi=150)
    plt.close()

    logger.info("Error bars on experiment_5a.png are standard deviations")
    logger.info(f"Saved {summary_path}")
    logger.info(f"Saved {os.path.join(OUTPUT_DIR, 'experiment_5a.png')}")
    logger.info(f"Saved {os.path.join(OUTPUT_DIR, 'experiment_5a_roc.png')}")
    logger.info(f"Saved {os.path.join(OUTPUT_DIR, f'experiment_5{section}.txt')}")


### cd pa3 && python experiment_5.py
### cd pa3 && /opt/anaconda3/bin/python experiment_5.py
if __name__ == "__main__":
    experiment_a()
