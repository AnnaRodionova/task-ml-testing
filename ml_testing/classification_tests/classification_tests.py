from sklearn.metrics import f1_score, roc_auc_score


def test_roc_auc_score(y_train, y_train_pred, y_holdout, y_holdout_pred, average='macro', sample_weight=None,
                       max_fpr=None,
                       multi_class='raise', labels=None):
    """
    The Scikit-Learn roc auc score, see the full documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

    :param y_train: array-like of shape (n_samples,) or (n_samples, n_classes).
    Correct train target values.

    :param y_train_pred: array-like of shape (n_samples,) or (n_samples, n_classes).
        Target scores.

    :param y_holdout: array-like of shape (n_samples,) or (n_samples, n_classes).
        Correct holdout target values.

    :param y_holdout_pred: array-like of shape (n_samples,) or (n_samples, n_classes). Target scores.

    :param average: {'micro', 'macro'(default), 'samples', 'weighted'} or None.
        If ``None``, the scores for each class are returned. Otherwise,
        this determines the type of averaging performed on the data:

        'micro':
            Calculate metrics globally by considering each element of the label
            indicator matrix as a label.
        'macro':
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        'weighted':
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label).
        'samples':
            Calculate metrics for each instance, and find their average.

    :param sample_weight: array-like of shape (n_samples,), default=None
        Sample weights.

    :param max_fpr: float > 0 and <= 1, default=None

    :param multi_class: {'raise'(default), 'ovr', 'ovo'}
        'ovr':
            Computes the AUC of each class against the rest.
        'ovo':
            Computes the average AUC of all possible pairwise combinations of
            classes.

    :param labels: array-like of shape (n_classes,)

    :return: string
    """
    roc_auc_train = roc_auc_score(y_train,
                                  y_train_pred,
                                  average=average,
                                  sample_weight=sample_weight,
                                  max_fpr=max_fpr,
                                  multi_class=multi_class,
                                  labels=labels)
    roc_auc_holdout = roc_auc_score(y_holdout,
                                    y_holdout_pred,
                                    average=average,
                                    sample_weight=sample_weight,
                                    max_fpr=max_fpr,
                                    multi_class=multi_class,
                                    labels=labels)
    score = f"Train ROC AUC = {roc_auc_train:.3f}, Holdout ROC AUC = {roc_auc_holdout:.3f}"
    print(score)


def test_f1_score(y_train, y_train_pred, y_holdout, y_holdout_pred, labels=None, pos_label=1, average='binary',
                  sample_weight=None, zero_division='warn'):
    """
    The Scikit-Learn f1 score, see the full documentation here:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    :param y_train: 1d array-like, or label indicator array / sparse matrix
        Correct train target values.

    :param 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    :param y_holdout: 1d array-like, or label indicator array / sparse matrix
        Correct holdout target values.

    :param y_holdout_pred: 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.

    :param labels: list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    :param pos_label: str or int, 1 by default
        The class to report if average='binary' and the data is binary.  If
        the data are multiclass or multilabel, this will be ignored; setting
        labels=[pos_label] and average != 'binary' will report scores for
        that label only.

    :param average: string,
          [None, 'binary'(default), 'micro', 'macro', 'samples', 'weighted']
          This parameter is required for multiclass/multilabel targets.  If None,
          the scores for each class are returned.  Otherwise, this determines the
          type of averaging performed on the data.

          'binary':
             Only report results for the class specified by pos_label.  This is
             applicable only if targets (y_{true, pred}) are binary.
          'micro':
             Calculate metrics globally by counting the total true positives,
             false negatives and false positives.
          'macro':
             Calculate metrics for each label, and find their unweighted mean.
             This does not take label imbalance into account.
           'weighted':
             Calculate metrics for each label, and find their average weighted by
             support (the number of true instances for each label).  This alters
             'macro' to account for label imbalance; it can result in an F-score
             that isnot between precision and recall.
           'samples':
             Calculate metrics for each instance, and find their average (only
             meaningful for multilabel classification where this differs from
             accuracy_score).

    :param sample_weight: array-like of shape (n_samples,), default=None
        Sample weights.

    :param zero_division: "warn", 0 or 1, default="warn"
        Sets the value to return when there is a zero division, i.e. when all
        predictions and labels are negative. If set to "warn", this acts as 0,
        but warnings are also raised.

    :return: string
    """
    f1_score_train = f1_score(y_train,
                              y_train_pred,
                              labels=labels,
                              pos_label=pos_label,
                              average=average,
                              sample_weight=sample_weight,
                              zero_division=zero_division)

    f1_score_holdout = f1_score(y_holdout,
                                y_holdout_pred,
                                labels=labels,
                                pos_label=pos_label,
                                average=average,
                                sample_weight=sample_weight,
                                zero_division=zero_division)
    score = f"Train f1 = {f1_score_train:.3f}, Holdout f1 = {f1_score_holdout:.3f}"
    print(score)
