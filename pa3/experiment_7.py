import csv
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


from src.ann.activations.linear import Linear
from src.ann.activations.relu import ReLU
from src.ann.activations.sigmoid import Sigmoid
from src.ann.activations.tanh import Tanh
from src.ann.clf import ANNC
from src.ann.data.header import ContinuousFeature, GroundTruth, Header
from src.ann.optimizers.sgd import SGDOptimizer


OUTPUT_DIR = os.path.join("output", "experiment_7")
BINARY_SUMMARY = os.path.join("output", "experiment_5", "experiment_5b.csv")


def make_logger(section: str) -> logging.Logger:
    logger = logging.getLogger(f"experiment_7{section}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, f"experiment_7{section}.txt"), mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def sep(logger: logging.Logger):
    logger.info("-"*100)


def load_sklearn_classification(name: str):
    bunch = load_iris() if name == "iris" else load_digits()
    X = np.asarray(bunch.data, dtype=float)
    y = np.asarray(bunch.target).reshape(-1)
    header = Header()
    for feature_idx in range(X.shape[1]):
        header.add_feature(ContinuousFeature(f"x{feature_idx}", feature_idx))
    classes = [str(label) for label in np.unique(y)]
    header.add_ground_truth(GroundTruth(classes))
    return header, X, y


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[y.astype(int)]


def load_binary_curves():
    if not os.path.exists(BINARY_SUMMARY):
        return None
    curves = {}
    with open(BINARY_SUMMARY, newline="") as f:
        for row in csv.DictReader(f):
            dataset_curves = curves.setdefault(row["dataset"], {})
            activation_curve = dataset_curves.setdefault(row["activation"], {})
            activation_curve[int(row["iters"])] = float(row["aroc mean"])
    return curves


def strictly_increasing(values) -> bool:
    return all(later > earlier for earlier, later in zip(values, values[1:]))


def curve_means(activation_curves, iteration_values):
    return [activation_curves[n_iters] for n_iters in iteration_values]


def best_activation(dataset_curves, n_iters: int) -> str:
    return max(dataset_curves, key=lambda name: dataset_curves[name][n_iters])


def experiment_a():
    section = "a"
    logger = make_logger(section)
    num_hidden_units = 5
    iteration_values = [int(round(v)) for v in np.linspace(100, 10_000, 5)]
    lr = 1e-2
    epsilon = 0.0
    gamma = 0.01
    dataset_names = ["iris", "digits"]
    activations = {
        "linear": Linear,
        "sigmoid": Sigmoid,
        "relu": ReLU,
        "tanh": Tanh,
    }

    logger.info("Experiment (a):\n")
    logger.info(f"Model: {num_hidden_units} hidden units")
    logger.info("Hidden activations: Linear, Sigmoid, ReLU, Tanh")
    logger.info("Output: Softmax + categorical cross-entropy")
    logger.info("Labels: one-hot encoded outside the model with np.eye")
    logger.info(f"gamma: {gamma}")
    logger.info("lr: 0.01")
    logger.info("epsilon: 0")
    logger.info(f"Iterations: {', '.join(str(n) for n in iteration_values)}")
    logger.info("Folds: 5")
    logger.info("Seed: 12345")
    logger.info("Preprocessing: StandardScaler on continuous features only")
    logger.info("Init: np.random.seed(12345) before every model")
    logger.info("epsilon is 0 so each run uses its full iteration budget")
    logger.info("AROC: weighted one-vs-rest")

    binary_curves = load_binary_curves()
    if binary_curves is None:
        logger.info(f"No binary reference found at {BINARY_SUMMARY}")
    else:
        logger.info("Binary reference: experiment 5(b), 5 hidden units, gamma 0.01")
        for dataset_name, dataset_curves in binary_curves.items():
            iteration_grid = sorted(next(iter(dataset_curves.values())))
            rising = all(
                strictly_increasing(curve_means(activation_curve, iteration_grid))
                for activation_curve in dataset_curves.values()
            )
            winner = best_activation(dataset_curves, iteration_grid[-1])
            winner_mean = dataset_curves[winner][iteration_grid[-1]]
            trend = "rose at every iteration step" if rising else "did not rise at every iteration step"
            logger.info(
                f"{dataset_name}: AROC {trend} for every activation. "
                f"Best activation at {iteration_grid[-1]} iterations: {winner} "
                f"(mean AROC {winner_mean:.6f})"
            )

    results = {
        dataset_name: {
            activation_name: {n_iters: [] for n_iters in iteration_values}
            for activation_name in activations
        }
        for dataset_name in dataset_names
    }

    for dataset_name in dataset_names:
        sep(logger)
        logger.info(f"{dataset_name.capitalize()} dataset")
        header, X, y_gt = load_sklearn_classification(dataset_name)
        n_classes = len(header.ground_truth.domain())
        cont_idx = list(range(X.shape[1]))

        classes, counts = np.unique(y_gt, return_counts=True)
        logger.info(f"Shape: {X.shape}")
        logger.info(f"Class counts: {dict(zip(classes.tolist(), counts.tolist()))}")
        logger.info(f"Classes: {n_classes}")
        logger.info(f"Continuous features: {X.shape[1]}")
        logger.info("Discrete features: 0")
        logger.info(f"Network inputs: {X.shape[1]}")

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)
        for activation_name, activation_cls in activations.items():
            sep(logger)
            logger.info(f"Activation: {activation_name}")
            for n_iters in iteration_values:
                logger.info(f"Iterations: {n_iters}")
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_gt), start=1):
                    logger.info(f"Fold: {fold}")
                    logger.info(f"Train size: {len(train_idx)}")
                    logger.info(f"Test size: {len(test_idx)}")

                    X_train, X_test = X[train_idx].copy(), X[test_idx].copy()
                    y_train, y_test = y_gt[train_idx], y_gt[test_idx]

                    scaler = StandardScaler()
                    X_train[:, cont_idx] = scaler.fit_transform(X_train[:, cont_idx])
                    X_test[:, cont_idx] = scaler.transform(X_test[:, cont_idx])

                    np.random.seed(12345)
                    model = ANNC(header, num_hidden_units, activation_cls())
                    optimizer = SGDOptimizer(model.parameters(), lr)

                    t0 = time.perf_counter()
                    model.fit(
                        X_train,
                        one_hot(y_train, n_classes),
                        optimizer,
                        gamma=gamma,
                        epsilon=epsilon,
                        max_iters=n_iters,
                    )
                    elapsed = time.perf_counter() - t0

                    train_accuracy = accuracy_score(y_train, model.predict(X_train))
                    logger.info(f"Train time (s): {elapsed:.2f}")
                    logger.info(f"Train accuracy: {train_accuracy}")

                    y_score = model.predict_proba(X_test)
                    aroc = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
                    results[dataset_name][activation_name][n_iters].append(aroc)
                    logger.info(f"AROC: {aroc}")

                values = np.asarray(results[dataset_name][activation_name][n_iters])
                logger.info(f"Mean aroc: {values.mean()}")
                logger.info(f"Variance aroc: {values.var()}")

    sep(logger)
    logger.info("Summary (mean and variance of AROC across 5 folds):")
    header_row = f"{'dataset':<12}{'activation':<12}{'iters':>8}{'aroc mean':>14}{'aroc var':>14}"
    logger.info(header_row)
    for dataset_name in dataset_names:
        for activation_name in activations:
            for n_iters in iteration_values:
                values = np.asarray(results[dataset_name][activation_name][n_iters])
                logger.info(
                    f"{dataset_name:<12}{activation_name:<12}{n_iters:>8}"
                    f"{values.mean():>14.6f}{values.var():>14.6f}"
                )

    summary_path = os.path.join(OUTPUT_DIR, "experiment_7a.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "activation", "iters", "aroc mean", "aroc var"])
        for dataset_name in dataset_names:
            for activation_name in activations:
                for n_iters in iteration_values:
                    values = np.asarray(results[dataset_name][activation_name][n_iters])
                    writer.writerow([
                        dataset_name,
                        activation_name,
                        n_iters,
                        f"{values.mean():.6f}",
                        f"{values.var():.6f}",
                    ])

    fig, axes = plt.subplots(1, len(dataset_names), figsize=(10, 4.5), sharey=True)
    for ax, dataset_name in zip(axes, dataset_names):
        for activation_name in activations:
            means = [
                np.mean(results[dataset_name][activation_name][n_iters])
                for n_iters in iteration_values
            ]
            stds = [
                np.std(results[dataset_name][activation_name][n_iters])
                for n_iters in iteration_values
            ]
            ax.errorbar(iteration_values, means, yerr=stds, capsize=3, marker="o", label=activation_name)
        ax.set_title(dataset_name)
        ax.set_xlabel("Training iterations")
        ax.set_ylim(0, 1.05)
        ax.legend()
    axes[0].set_ylabel("AROC")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "experiment_7a.png"), dpi=150)
    plt.close(fig)

    sep(logger)
    logger.info("Comparison with binary experiment 5(b):")
    max_iters = iteration_values[-1]
    best_at_max = {}
    rose = {}
    for dataset_name in dataset_names:
        means = {
            activation_name: [
                float(np.mean(results[dataset_name][activation_name][n_iters]))
                for n_iters in iteration_values
            ]
            for activation_name in activations
        }
        rose[dataset_name] = all(strictly_increasing(values) for values in means.values())
        best_name = max(means, key=lambda name: means[name][-1])
        best_at_max[dataset_name] = best_name
        trend = "rose at every iteration step" if rose[dataset_name] else "did not rise at every iteration step"
        logger.info(
            f"{dataset_name}: AROC {trend} for every activation. "
            f"Best activation at {max_iters} iterations: {best_name} "
            f"(mean AROC {means[best_name][-1]:.6f})"
        )
        for activation_name in activations:
            for n_iters, mean in zip(iteration_values, means[activation_name]):
                logger.info(f"{dataset_name} {activation_name} @ {n_iters}: {mean:.6f}")

    multiclass_rose = all(rose.values())
    multiclass_winners = set(best_at_max.values())
    if binary_curves is None:
        logger.info("Binary experiment 5(b) was not available, so the trend comparison is unanswered.")
    else:
        binary_rose = {}
        binary_winners = {}
        for dataset_name, dataset_curves in binary_curves.items():
            iteration_grid = sorted(next(iter(dataset_curves.values())))
            binary_rose[dataset_name] = all(
                strictly_increasing(curve_means(activation_curve, iteration_grid))
                for activation_curve in dataset_curves.values()
            )
            binary_winners[dataset_name] = best_activation(dataset_curves, iteration_grid[-1])
        binary_all_rose = all(binary_rose.values())
        if multiclass_rose and binary_all_rose:
            logger.info(
                "Same trend as the binary experiments: AROC rose with the number of training iterations "
                "on every dataset and every activation."
            )
        else:
            logger.info(
                "The iteration trend differs from the binary experiments. "
                f"Binary AROC rose on every dataset: {binary_all_rose}. "
                f"Multiclass AROC rose on every dataset: {multiclass_rose}."
            )
        if len(multiclass_winners) == 1 and set(binary_winners.values()) == multiclass_winners:
            winner = multiclass_winners.pop()
            logger.info(
                f"Same best activation as the binary experiments at {max_iters} iterations: {winner}."
            )
        else:
            binary_summary = ", ".join(
                f"{dataset_name} {activation_name}" for dataset_name, activation_name in binary_winners.items()
            )
            multiclass_summary = ", ".join(
                f"{dataset_name} {activation_name}" for dataset_name, activation_name in best_at_max.items()
            )
            logger.info(f"Best binary activations at {max_iters} iterations: {binary_summary}.")
            logger.info(f"Best multiclass activations at {max_iters} iterations: {multiclass_summary}.")

    logger.info("Error bars on experiment_7a.png are standard deviations")
    logger.info(f"Saved {summary_path}")
    logger.info(f"Saved {os.path.join(OUTPUT_DIR, 'experiment_7a.png')}")
    logger.info(f"Saved {os.path.join(OUTPUT_DIR, f'experiment_7{section}.txt')}")


### cd pa3 && python experiment_7.py
### cd pa3 && /opt/anaconda3/bin/python experiment_7.py
if __name__ == "__main__":
    experiment_a()
