from sklearn.metrics import mean_squared_error, r2_score


def test_r2_score(y_train, y_train_pred, y_holdout, y_holdout_pred, sample_weight=None,
                  multioutput='uniform_average'):
    """
    The Scikit-Learn r2 score, see the full documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html

    :param y_train: array-like of shape (n_samples,) or (n_samples, n_classes).
    Correct train target values.

    :param y_train_pred: array-like of shape (n_samples,) or (n_samples, n_classes).
        Target scores.

    :param y_holdout: array-like of shape (n_samples,) or (n_samples, n_classes).
        Correct holdout target values.

    :param y_holdout_pred: array-like of shape (n_samples,) or (n_samples, n_classes).
        Target scores.

    :param sample_weight: array-like of shape (n_samples,), optional
        Sample weights.

    :param multioutput: string in ['raw_values', 'uniform_average', \
        'variance_weighted'] or None or array-like of shape (n_outputs)
        Defines aggregating of multiple output scores.
        Array-like value defines weights used to average scores.
        Default is "uniform_average".
        'raw_values' :
            Returns a full set of scores in case of multioutput input.
        'uniform_average' :
            Scores of all outputs are averaged with uniform weight.
        'variance_weighted' :
            Scores of all outputs are averaged, weighted by the variances
            of each individual output.

    :return: string
    """
    r2_train = r2_score(y_train,
                        y_train_pred,
                        sample_weight=sample_weight,
                        multioutput=multioutput)

    r2_holdout = r2_score(y_holdout,
                          y_holdout_pred,
                          sample_weight=sample_weight,
                          multioutput=multioutput)
    print(
        f"Train Coefficient of determination = {r2_train:.3f}, Holdout Coefficient of determination = {r2_holdout:.3f}")


def test_rmse(y_train, y_train_pred, y_holdout, y_holdout_pred, sample_weight=None, multioutput='uniform_average'):
    """
    The Scikit-Learn RMSE score, see the full documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

    :param y_train: array-like of shape (n_samples,) or (n_samples, n_classes).
    Correct train target values.

    :param y_train_pred: array-like of shape (n_samples,) or (n_samples, n_classes).
        Target scores.

    :param y_holdout: array-like of shape (n_samples,) or (n_samples, n_classes).
        Correct holdout target values.

    :param y_holdout_pred: array-like of shape (n_samples,) or (n_samples, n_classes).
        Target scores.

    :param sample_weight: array-like of shape (n_samples,), optional
        Sample weights.

    :param multioutput: string in ['raw_values', 'uniform_average']
        or array-like of shape (n_outputs)
        Defines aggregating of multiple output values.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.

    :return: string
    """
    rmse_train = mean_squared_error(y_train,
                                    y_train_pred,
                                    sample_weight=sample_weight,
                                    multioutput=multioutput,
                                    squared=False)

    rmse_holdout = mean_squared_error(y_holdout,
                                      y_holdout_pred,
                                      sample_weight=sample_weight,
                                      multioutput=multioutput,
                                      squared=False)

    print(f"Train RMSE = {rmse_train:.3f}, Holdout RMSE = {rmse_holdout:.3f}")
