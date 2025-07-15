import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from typing import List, Tuple

from utils.general_utils import save_dataframe


# --- GENERIC FUNCTIONS ---


# define generic function to evaluate a predictive method (such as LogisticRegression)
def evaluate_predictive_method(
        df_train,
        df_test,
        features_cols,
        response_col,
        predictive_method=LogisticRegression,
        return_trained_method=False,
        compute_accuracy=False,
        trained_method=None,
        binary=True,
        return_brier_average=True,
        **kwargs,
):
    if len(features_cols) == 1:
        if type(df_train[features_cols[0]].iloc[0]) == list:
            # todo check if this is ever used
            df_train[features_cols[0]] = df_train[features_cols[0]].apply(
                lambda x: np.array(x)
            )
            df_test[features_cols[0]] = df_test[features_cols[0]].apply(
                lambda x: np.array(x)
            )
        if type(df_train[features_cols[0]].iloc[0]) == np.ndarray:
            # if the features are already a numpy array, then don't convert to numpy array
            features_train = np.array(list(df_train[features_cols[0]].values))
            features_test = np.array(list(df_test[features_cols[0]].values))
        else:
            # features are not a list or a numpy array, so assume it is a single number
            # traditional one
            features_train = df_train[features_cols].to_numpy()
            features_test = df_test[features_cols].to_numpy()
    else:
        # traditional one
        features_train = df_train[features_cols].to_numpy()
        features_test = df_test[features_cols].to_numpy()

    if response_col in features_cols:
        raise ValueError("response_col must not be in features_cols")

    labels_train = df_train[response_col]
    labels_test = df_test[response_col]

    return _evaluate_predictive_method_from_arrays(
        features_train,
        labels_train,
        features_test,
        labels_test,
        predictive_method,
        return_trained_method,
        compute_accuracy,
        trained_method,
        binary=binary,
        return_brier_average=return_brier_average,
        **kwargs,
    )


def _evaluate_predictive_method_from_arrays(
        features_train,
        labels_train,
        features_test,
        labels_test,
        predictive_method=LogisticRegression,
        return_trained_method=False,
        compute_accuracy=False,
        trained_method=None,
        binary=True,
        return_brier_average=True,
        **kwargs,
):
    if trained_method is not None:
        method_instance = trained_method
    else:
        # fit logistic regression using training features and the agent col as response
        method_instance = predictive_method(**kwargs)
        method_instance.fit(features_train, labels_train)

    return_dict = {}
    if binary:
        # evaluate on the test set
        y_pred = method_instance.predict_proba(features_test)[:, 1]
        y_pred_train = method_instance.predict_proba(features_train)[:, 1]

        BrierScore, Calibration, Refinement = brierDecomp(
            y_pred, labels_test, return_brier_average=return_brier_average
        )
        winkler_score = compute_winklers_score(y_pred, labels_test)
        # compute the ROC AUC using sklearn
        if not (sum(labels_test) == 0 or sum(labels_test) == len(labels_test)):
            roc_auc = roc_auc_score(labels_test, y_pred)
        else:
            roc_auc = np.nan

        if compute_accuracy:
            # compute accuracy by thresholding at 0.5
            y_pred_binary = y_pred > 0.5
            accuracy = np.mean(y_pred_binary == labels_test)

        return_dict["BrierScore"] = BrierScore
        return_dict["WinklerScore"] = winkler_score
        return_dict["Calibration"] = Calibration
        return_dict["Refinement"] = Refinement
        return_dict["roc_auc"] = roc_auc
    else:
        y_pred = method_instance.predict(features_test)
        y_pred_train = method_instance.predict(features_train)
        accuracy = np.mean(y_pred == labels_test)

    if compute_accuracy:
        return_dict["accuracy"] = accuracy
    if return_trained_method:
        return_dict["trained_method"] = method_instance

    return_dict["predictions_test"] = y_pred
    return_dict["predictions_train"] = y_pred_train
    return_dict["arc_test"] = arc_points(labels_test, y_pred)
    return_dict["arc_train"] = arc_points(labels_train, y_pred_train)

    return return_dict


# define generic function to evaluate a predictive method (such as LogisticRegression)
def evaluate_regression_method(
        df_train,
        df_test,
        features_cols,
        response_col,
        predictive_method=LinearRegression,
        return_trained_method=False,
        trained_method=None,
        return_brier_average=True,
        **kwargs,
):
    if (
            len(features_cols) == 1
            and type(df_train[features_cols[0]].iloc[0]) == np.ndarray
    ):
        # if the features are already a numpy array, then don't convert to numpy array
        features_train = np.array(list(df_train[features_cols[0]].values))
        features_test = np.array(list(df_test[features_cols[0]].values))
    else:
        # traditional one
        features_train = df_train[features_cols].to_numpy()
        features_test = df_test[features_cols].to_numpy()

    if response_col in features_cols:
        raise ValueError("response_col must not be in features_cols")

    if trained_method is not None:
        method_instance = trained_method
    else:
        # fit logistic regression using training features and the agent col as response
        method_instance = predictive_method(**kwargs)
        method_instance.fit(features_train, df_train[response_col])

    # evaluate on the test set
    y_pred = method_instance.predict(features_test)

    # Brier Score essentially correponds to the MSE for regression
    BrierScore, Calibration, Refinement = brierDecomp(
        y_pred, df_test[response_col], return_brier_average=return_brier_average
    )

    return_list = [BrierScore, Calibration, Refinement]
    if return_trained_method:
        return_list.append(method_instance)
    return return_list


def evaluate_non_probabilistic_predictive_method(
        df_train,
        df_test,
        features_cols,
        response_col,
        predictive_method=DecisionTreeClassifier,
        return_trained_method=False,
        trained_method=None,
        **kwargs,
):
    features_train = df_train[features_cols].to_numpy()
    features_test = df_test[features_cols].to_numpy()

    if trained_method is not None:
        method_instance = trained_method
    else:
        # fit logistic regression using training features and the agent col as response
        method_instance = predictive_method(**kwargs)
        method_instance.fit(features_train, df_train[response_col])

    # evaluate on the test set
    y_pred_binary = method_instance.predict(features_test)

    accuracy = np.mean(y_pred_binary == df_test[response_col])

    return_list = [accuracy]
    if return_trained_method:
        return_list.append(method_instance)
    return return_list


def brierScore(preds, outs):
    return 1 / len(preds) * sum((preds - outs) ** 2)


def brierDecomp(preds, outs, return_brier_average=True):
    brier = (preds - outs) ** 2
    if return_brier_average:
        brier = 1 / len(preds) * sum(brier)
    ## bin predictions
    bins = np.linspace(0, 1, 11)
    binCenters = (bins[:-1] + bins[1:]) / 2
    binPredInds = np.digitize(preds, binCenters)
    binnedPreds = bins[binPredInds]

    binTrueFreqs = np.zeros(10)
    binPredFreqs = np.zeros(10)
    binCounts = np.zeros(10)

    for i in range(10):
        idx = (preds >= bins[i]) & (preds < bins[i + 1])

        binTrueFreqs[i] = np.sum(outs[idx]) / np.sum(idx) if np.sum(idx) > 0 else 0
        # print(np.sum(outs[idx]), np.sum(idx), binTrueFreqs[i])
        binPredFreqs[i] = np.mean(preds[idx]) if np.sum(idx) > 0 else 0
        binCounts[i] = np.sum(idx)

    calibration = (
        np.sum(binCounts * (binTrueFreqs - binPredFreqs) ** 2) / np.sum(binCounts)
        if np.sum(binCounts) > 0
        else 0
    )
    refinement = (
        np.sum(binCounts * (binTrueFreqs * (1 - binTrueFreqs))) / np.sum(binCounts)
        if np.sum(binCounts) > 0
        else 0
    )
    # Compute refinement component
    # refinement = brier - calibration
    return brier, calibration, refinement


def arc_points(
        labels: List[float], predictions: List[float], num_points: int = 100
) -> List[Tuple[float, float]]:
    assert len(labels) == len(
        predictions
    ), f"Expected the same number of labels and predictions, got {len(labels)}, {len(predictions)} respectively"
    predictions_series = pd.Series(sorted(predictions))
    rejection_rates = [i / (num_points - 1) for i in range(num_points)]
    out: List[Tuple[float, float]] = []
    for rejection_rate in rejection_rates:
        threshold = predictions_series.quantile(rejection_rate)
        predictions_above_threshold = [
            label
            for label, prediction in zip(labels, predictions) if prediction > threshold
        ]
        if len(predictions_above_threshold) == 0:
            # If rejection rate is 1, no predictions will be above the threshold and accuracy will be 0 / 0
            # So hardcode the final point
            out.append((1, 1))
            continue
        accuracy = (
                sum(
                    predictions_above_threshold
                )
                / len(
            predictions_above_threshold
        )
        )
        out.append((rejection_rate, accuracy))
    return out


def compute_winklers_score(preds, outs):
    """
    Compute Winkler's Score as described by:
    Winkler, R. L. (1994).
    "Evaluating probabilities: Asymmetric scoring rules."
    Management Science,923 40(11):1395–1405

    Parameters
    ----------
    preds : array-like
        An array of predicted probabilities a(x_i) for each instance i.
        Each element should be in [0,1].
    outs : array-like
        An array of ground-truth binary outcomes v_i (0 or 1).

    Returns
    -------
    float
        The Winkler's Score.
    """
    # Convert inputs to numpy arrays
    preds = np.array(preds, dtype=float)
    outs = np.array(outs, dtype=int)

    # Check dimensions
    if preds.shape[0] != outs.shape[0]:
        raise ValueError("Predictions and ground_truth must have the same length.")

    n = len(preds)

    # Compute c_j as the average success rate (fraction of positives)
    c_j = np.mean(outs)

    # c_j must be in (0,1) for Winkler's score to be well-defined.
    # If c_j = 0 or c_j = 1, the transformation breaks down.
    if c_j <= 0.0 or c_j >= 1.0:
        print("The computed c_j is not strictly between 0 and 1. "
              "Ensure that ground_truth has at least one positive and one negative outcome.")
        return np.nan

    # Indicator for v_i
    v_is_one = (outs == 1).astype(float)
    v_is_zero = (outs == 0).astype(float)

    # Indicators for prediction relative to c_j
    pred_leq_c = (preds <= c_j).astype(float)
    pred_gt_c = (preds > c_j).astype(float)

    # Compute alpha_i:
    # alpha_i = [(1-c_j)^2 - (1-a_i)^2]*1{v_i=1} + (c_j^2 - a_i^2)*1{v_i=0}
    term_v1 = ((1 - c_j) ** 2 - (1 - preds) ** 2) * v_is_one
    term_v0 = (c_j ** 2 - preds ** 2) * v_is_zero
    alpha = term_v1 + term_v0

    # Compute beta_i:
    # beta_i = c_j^2 * 1{a_i ≤ c_j} + (1-c_j)^2 * 1{a_i > c_j}
    beta = c_j ** 2 * pred_leq_c + (1 - c_j) ** 2 * pred_gt_c

    # Compute Winkler's Score = (1/n) * sum(alpha_i / beta_i)
    score = np.mean(alpha / beta)

    return score


def extract_single_column_features(df, feature_column):
    if type(df[feature_column].iloc[0]) == list:
        # todo check if this is ever used
        df[feature_column] = df[feature_column].apply(lambda x: np.array(x))
    if type(df[feature_column].iloc[0]) == np.ndarray:
        # if the features are already a numpy array, then don't convert to numpy array
        features = np.array(list(df[feature_column].values))
    else:
        # features are not a list or a numpy array, so assume it is a single number
        features = df[feature_column].to_numpy()
    return features


predictive_method_list = [
    (LogisticRegression, {}, "logistic_regression_l2"),
    (
        LogisticRegression,
        {"penalty": "l1", "solver": "liblinear"},
        "logistic_regression_l1_c=1",
    ),
    (
        LogisticRegression,
        {"penalty": "l1", "solver": "liblinear", "C": 0.1},
        "logistic_regression_l1_c=0.1",
    ),
    (XGBClassifier, {}, "xgboost"),
]  # , (LinearSVC, {}, "linear_svc_l2"), (LinearSVC, {"penalty": "l1", "dual": False}, "linear_svc_l1_c=1"), (LinearSVC, {"penalty": "l1", "dual": False, "C": 0.1}, "linear_svc_l1_c=0.1")]


# ----------------------------------

def _check_skip(res_df, pred_method_name, feature_name, llm):
    if (
            len(res_df) > 0
            and len(
        res_df[
            (res_df["predictive_method"] == pred_method_name)
            & (res_df["features"] == feature_name)
            & (res_df["llm"] == llm)
        ]
    )
            > 0
    ):
        print(
            f"Skipping {feature_name}, {pred_method_name} for {llm} because it is already in the dataframe"
        )
        return True
    else:
        print(f"Doing {feature_name}, {pred_method_name} for {llm}")
        return False


def _concat_and_save(
        res_df,
        pred_method_name,
        feature_name,
        llm,
        BrierScore_val,
        WinklerScore_val,
        Calibration_val,
        Refinement_val,
        roc_auc_val,
        BrierScore_test,
        WinklerScore_test,
        Calibration_test,
        Refinement_test,
        roc_auc_test,
        predictions_train,
        predictions_val,
        predictions_test,
        arc_train,
        arc_test,
        arc_val,
        llm_accuracy_train,
        llm_accuracy_val,
        llm_accuracy_test,
        trained_method,
        filename,
):
    res_df = pd.concat(
        [
            res_df,
            pd.DataFrame(
                {
                    "predictive_method": pred_method_name,
                    "features": feature_name,
                    "llm": llm,
                    "BrierScore_val": BrierScore_val,
                    "WinklerScore_val": WinklerScore_val,
                    "Calibration_val": Calibration_val,
                    "Refinement_val": Refinement_val,
                    "AUROC_val": roc_auc_val,
                    "BrierScore_test": BrierScore_test,
                    "WinklerScore_test": WinklerScore_test,
                    "Calibration_test": Calibration_test,
                    "Refinement_test": Refinement_test,
                    "AUROC_test": roc_auc_test,
                    "predictions_train": [predictions_train.tolist()],
                    "predictions_val": [predictions_val.tolist()],
                    "predictions_test": [predictions_test.tolist()],
                    "arc_train": [arc_train],
                    "arc_test": [arc_test],
                    "arc_val": [arc_val],
                    "llm_accuracy_train": llm_accuracy_train,
                    "llm_accuracy_val": llm_accuracy_val,
                    "llm_accuracy_test": llm_accuracy_test,
                    "trained_classifier": trained_method,
                },
                index=[0],
            ),
        ]
    )

    save_dataframe(filename, res_df)
    return res_df


def evaluate_and_update(
        res_df,
        train_df,
        validation_df,
        test_df,
        features,
        predictive_method,
        pred_method_name,
        feature_name,
        llm,
        filename,
        **kwargs,
):
    if not _check_skip(res_df, pred_method_name, feature_name, llm):
        results_dict_test = evaluate_predictive_method(
            train_df,
            test_df,
            features,
            f"Success_{llm}",
            predictive_method=predictive_method,
            return_trained_method=True,
            **kwargs,
        )
        # unpack results:
        BrierScore_test = results_dict_test["BrierScore"]
        WinklerScore_test = results_dict_test["WinklerScore"]
        Calibration_test = results_dict_test["Calibration"]
        Refinement_test = results_dict_test["Refinement"]
        roc_auc_test = results_dict_test["roc_auc"]
        trained_method = results_dict_test["trained_method"]
        predictions_train = results_dict_test["predictions_train"]
        predictions_test = results_dict_test["predictions_test"]
        arc_test = results_dict_test["arc_test"]
        arc_train = results_dict_test["arc_train"]

        results_dict_val = evaluate_predictive_method(
            train_df,
            validation_df,
            features,
            f"Success_{llm}",
            predictive_method=predictive_method,
            trained_method=trained_method,
            **kwargs,
        )

        BrierScore_val = results_dict_val["BrierScore"]
        WinklerScore_val = results_dict_val["WinklerScore"]
        Calibration_val = results_dict_val["Calibration"]
        Refinement_val = results_dict_val["Refinement"]
        roc_auc_val = results_dict_val["roc_auc"]
        predictions_val = results_dict_val["predictions_test"]
        arc_val = results_dict_val["arc_test"]

        llm_accuracy_train = train_df[f"Success_{llm}"].mean()
        llm_accuracy_val = validation_df[f"Success_{llm}"].mean()
        llm_accuracy_test = test_df[f"Success_{llm}"].mean()

        res_df = _concat_and_save(
            res_df,
            pred_method_name,
            feature_name,
            llm,
            BrierScore_val,
            WinklerScore_val,
            Calibration_val,
            Refinement_val,
            roc_auc_val,
            BrierScore_test,
            WinklerScore_test,
            Calibration_test,
            Refinement_test,
            roc_auc_test,
            predictions_train,
            predictions_val,
            predictions_test,
            arc_train,
            arc_test,
            arc_val,
            llm_accuracy_train,
            llm_accuracy_val,
            llm_accuracy_test,
            trained_method,
            filename,
        )
    return res_df


def evaluate_and_update_arrays(
        res_df,
        X_train,
        train_labels,
        X_val,
        val_labels,
        X_test,
        test_labels,
        predictive_method,
        pred_method_name,
        feature_name,
        llm,
        filename,
        **kwargs,
):
    if not _check_skip(res_df, pred_method_name, feature_name, llm):
        results_dict_test = _evaluate_predictive_method_from_arrays(
            X_train,
            train_labels,
            X_test,
            test_labels,
            predictive_method=predictive_method,
            return_trained_method=True,
            **kwargs,
        )

        # unpack results:
        BrierScore_test = results_dict_test["BrierScore"]
        WinklerScore_test = results_dict_test["WinklerScore"]
        Calibration_test = results_dict_test["Calibration"]
        Refinement_test = results_dict_test["Refinement"]
        roc_auc_test = results_dict_test["roc_auc"]
        trained_method = results_dict_test["trained_method"]
        predictions_train = results_dict_test["predictions_train"]
        predictions_test = results_dict_test["predictions_test"]
        arc_test = results_dict_test["arc_test"]
        arc_train = results_dict_test["arc_train"]

        results_dict_val = _evaluate_predictive_method_from_arrays(
            X_train,
            train_labels,
            X_val,
            val_labels,
            predictive_method=predictive_method,
            trained_method=trained_method,
            **kwargs,
        )

        BrierScore_val = results_dict_val["BrierScore"]
        WinklerScore_val = results_dict_val["WinklerScore"]
        Calibration_val = results_dict_val["Calibration"]
        Refinement_val = results_dict_val["Refinement"]
        roc_auc_val = results_dict_val["roc_auc"]
        predictions_val = results_dict_val["predictions_test"]
        arc_val = results_dict_val["arc_test"]

        llm_accuracy_train = train_labels.mean()
        llm_accuracy_val = val_labels.mean()
        llm_accuracy_test = test_labels.mean()

        res_df = _concat_and_save(
            res_df,
            pred_method_name,
            feature_name,
            llm,
            BrierScore_val,
            WinklerScore_val,
            Calibration_val,
            Refinement_val,
            roc_auc_val,
            BrierScore_test,
            WinklerScore_test,
            Calibration_test,
            Refinement_test,
            roc_auc_test,
            predictions_train,
            predictions_val,
            predictions_test,
            arc_train,
            arc_test,
            arc_val,
            llm_accuracy_train,
            llm_accuracy_val,
            llm_accuracy_test,
            trained_method,
            filename,
        )
    return res_df
