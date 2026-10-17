"""
Run KNN classification on the same cached feature vectors used by the SVM.

Each cache contains:
    X: one row per CSI window, 33 extracted features
    y: class label for each window

The evaluation uses stratified 5-fold cross-validation with a StandardScaler
and an inner training-fold search over a small set of K values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Experiment:
    name: str
    cache_path: Path
    class0: str
    class1: str


EXPERIMENTS = [
    Experiment(
        name="Empty Room vs Video Game",
        cache_path=Path("svm_empty_vs_video_features.npz"),
        class0="Empty Room",
        class1="Video Game",
    ),
    Experiment(
        name="5GHz Person LOS vs 5GHz Person",
        cache_path=Path("svm_5g_off_person_los_vs_person_features.npz"),
        class0="5GHz Person LOS",
        class1="5GHz Person",
    ),
    Experiment(
        name="5GHz Person LOS vs 5GHz Motion LOS",
        cache_path=Path("svm_5g_off_person_los_vs_motion_los_features.npz"),
        class0="5GHz Person LOS",
        class1="5GHz Motion LOS",
    ),
]


def evaluate_experiment(experiment: Experiment, seed: int = 42) -> dict[str, object]:
    data = np.load(experiment.cache_path, allow_pickle=False)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=int)

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_accuracies: list[float] = []
    best_ks: list[int] = []
    confusion_matrices: list[np.ndarray] = []

    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        max_k = min(15, len(train_index) - 1)
        candidate_ks = [k for k in (1, 3, 5, 7, 9, 11, 13, 15) if k <= max_k]
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier()),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid={
                "knn__n_neighbors": candidate_ks,
                "knn__weights": ["uniform", "distance"],
                "knn__metric": ["euclidean"],
            },
            cv=inner_cv,
            scoring="accuracy",
        )
        search.fit(X_train, y_train)
        prediction = search.predict(X_test)

        fold_accuracies.append(float(accuracy_score(y_test, prediction)))
        best_ks.append(int(search.best_params_["knn__n_neighbors"]))
        confusion_matrices.append(confusion_matrix(y_test, prediction, labels=[0, 1]))

    total_cm = np.sum(confusion_matrices, axis=0)
    return {
        "experiment": experiment.name,
        "samples": len(y),
        "class0": experiment.class0,
        "class1": experiment.class1,
        "class0_samples": int(np.sum(y == 0)),
        "class1_samples": int(np.sum(y == 1)),
        "mean_accuracy": float(np.mean(fold_accuracies)),
        "std_accuracy": float(np.std(fold_accuracies)),
        "best_k_mode": int(pd.Series(best_ks).mode().iloc[0]),
        "best_k_per_fold": ",".join(str(k) for k in best_ks),
        "true0_pred0": int(total_cm[0, 0]),
        "true0_pred1": int(total_cm[0, 1]),
        "true1_pred0": int(total_cm[1, 0]),
        "true1_pred1": int(total_cm[1, 1]),
    }


def main() -> None:
    rows = [evaluate_experiment(experiment) for experiment in EXPERIMENTS]
    output_path = Path("knn_feature_results.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"Saved KNN summary: {output_path}")
    for row in rows:
        print(
            f"{row['experiment']}: "
            f"accuracy={row['mean_accuracy']:.4f} +/- {row['std_accuracy']:.4f}, "
            f"mode k={row['best_k_mode']}, "
            f"samples={row['class0_samples']}+{row['class1_samples']}"
        )
        print(
            f"  {row['class0']} correct={row['true0_pred0']}/{row['class0_samples']} | "
            f"{row['class1']} correct={row['true1_pred1']}/{row['class1_samples']}"
        )


if __name__ == "__main__":
    main()
