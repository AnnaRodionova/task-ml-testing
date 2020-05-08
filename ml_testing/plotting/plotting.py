import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


def plot_roc_auc_binary(y_train, y_train_predict, y_holdout, y_holdout_predict):
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_predict)
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_holdout, tpr_holdout, _ = roc_curve(y_holdout, y_holdout_predict)
    roc_auc_holdout = auc(fpr_holdout, tpr_holdout)

    plt.figure()
    plt.plot(fpr_train, tpr_train, label='ROC curve train(area = %0.2f)' % roc_auc_train)
    plt.plot(fpr_holdout, tpr_holdout, label='ROC curve holdout(area = %0.2f)' % roc_auc_holdout)
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return plt


def plot_regression(x_holdout, y_holdout, y_pred, title):
    plt.scatter(x_holdout, y_holdout)
    plt.plot(x_holdout, y_pred, color='black')

    plt.title(label=title, fontdict={'fontsize': 15})
    plt.xlabel('x')
    plt.ylabel('y')
    return plt


def plot_regression_test(x_train, y_train, y_train_prediction, x_holdout, y_holdout, y_holdout_prediction):
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plot_regression(x_train, y_train, y_train_prediction, 'Train')

    plt.subplot(2, 2, 2)
    plot_regression(x_holdout, y_holdout, y_holdout_prediction, 'Holdout')
    return plt
