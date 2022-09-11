from utils import operate_dict, operate_dict_const
from embeddings import retrieve_embeddings

#basic
from tqdm import tqdm
import re

#scientific computing
import numpy as np
from scipy.interpolate import interp1d

#plotting
import matplotlib.pyplot as plt

#scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay



def evaluate_embedding(dimensions, walk_length, xtrain, ytrain, DIR, num_cv=3):

  #embed the training set
  filename = '_embeddings_dim:'+str(dimensions)+'_len:'+str(walk_length)
  embeddings = retrieve_embeddings(DIR+filename)
  xtrain = np.asarray([np.asarray([embeddings[row[0]] + embeddings[row[1]]]) for row in xtrain])
  xtrain = np.reshape(xtrain, (len(xtrain), dimensions))

  #use of logisitic regression with balanced weights for classification
  classifier = LogisticRegression(class_weight="balanced", max_iter=1600)
  
  cv = StratifiedKFold(n_splits=num_cv)
  reports = []
  for (train, test) in tqdm(cv.split(xtrain, ytrain)):

      #
      classifier.fit(xtrain[train], ytrain[train])

      #predictions
      ypred = classifier.predict(xtrain[test])

      #classification report
      report = classification_report(ytrain[test], ypred, output_dict=True, target_names=['no link','link'])
      pred_prob = classifier.predict_proba(xtrain[test])

      #auroc
      fpr, tpr, thresholds = roc_curve(ytrain[test], pred_prob[:,1], pos_label=1)
      auroc = auc(fpr, tpr)
      report['auroc_score'] = auroc
      reports.append(report)

      means = operate_dict_const(operate_dict(reports, lambda a,b: a+b), lambda a,b:a/b, len(reports))
      squares = [operate_dict_const(report, lambda a,_:a**2, 0) for report in reports]
      sum_squares = operate_dict_const(operate_dict(squares, lambda a,b: a+b), lambda a,_: a/len(reports), 0)
      variance = operate_dict([sum_squares, operate_dict_const(means, lambda a,_: a**2, 0)], lambda a,b: a-b)
      std_error = operate_dict_const(variance, lambda a,_: np.sqrt(a)/np.sqrt(len(reports)), 0)

  return {'means':means, 'std_error':std_error}


def plot_ROC(data, path, cv_num=6, title="", figsize=(7,7)):
    
    scores = {}
    fig, ax = plt.subplots(figsize=figsize)
    for (X,y, classifier, name, color) in tqdm(data):
        cv = StratifiedKFold(n_splits=cv_num)
        # classifier = Classifier(random_state=0)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100000)
        
        reports = []
        for i, (train, test) in tqdm(enumerate(cv.split(X, y))):
            classifier.fit(X[train], y[train])
            y_pred = classifier.predict_proba(X[test])[:, 1]
            viz = RocCurveDisplay.from_predictions(
                y[test], 
                y_pred,
                name="ROC fold {}".format(i),
                alpha=0.0,
                lw=1,
                ax=ax,
                color=color
            )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)


            #predictions
            ypred = classifier.predict(X[test])

            #classification report
            report = classification_report(y[test], ypred, output_dict=True, target_names=['no link','link'])
            pred_prob = classifier.predict_proba(X[test])

            #auroc
            fpr, tpr, thresholds = roc_curve(y[test], pred_prob[:,1], pos_label=1)
            auroc = auc(fpr, tpr)
            report['auroc_score'] = auroc
            reports.append(report)


        means = operate_dict_const(operate_dict(reports, lambda a,b: a+b), lambda a,b:a/b, len(reports))
        squares = [operate_dict_const(report, lambda a,_:a**2, 0) for report in reports]
        sum_squares = operate_dict_const(operate_dict(squares, lambda a,b: a+b), lambda a,_: a/len(reports), 0)
        variance = operate_dict([sum_squares, operate_dict_const(means, lambda a,_: a**2, 0)], lambda a,b: a-b)
        std_error = operate_dict_const(variance, lambda a,_: np.sqrt(a)/np.sqrt(len(reports)), 0)
        scores[name] = {'means':means, 'std_error':std_error}

        ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            label=r"Mean ROC of "+name+" (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc/np.sqrt(len(tprs))),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr/np.sqrt(len(tprs)), 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr/np.sqrt(len(tprs)), 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=color,
            alpha=0.2,
            label=r"$\pm$ 1 std. err. for "+name,
        )



    plt.gcf()
    handles, labels = plt.gca().get_legend_handles_labels()
    l = [(i,name) for i,name in enumerate(labels) if not re.search("^ROC", name)]
    handles = [handles[i] for (i,_) in l]
    labels = [name for (_,name) in l]
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title(title)
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.savefig(path)#glob_dir+'/ROCs_best_models', bbox_inches = 'tight')
    plt.show()

    return scores


def plot_PRcurve(data, path, cv_num=6, title="", figsize=(7,7)):
    
    scores = {}
    fig, ax = plt.subplots(figsize=figsize)
    for (X,y, classifier, name, color) in tqdm(data):
        cv = StratifiedKFold(n_splits=cv_num)
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 1000000)
        
        reports = []
        for i, (train, test) in tqdm(enumerate(cv.split(X, y))):
            classifier.fit(X[train], y[train])
            y_pred = classifier.predict_proba(X[test])[:, 1]
            viz = PrecisionRecallDisplay.from_predictions(
                y[test], 
                y_pred,
                name="ROC fold {}".format(i),
                alpha=0.0,
                lw=1,
                ax=ax,
                color=color
            )
            f = interp1d(viz.recall, viz.precision) #np.interp(mean_fpr, viz.recall, viz.precision)
            interp_tpr = f(mean_fpr) 
            tprs.append(interp_tpr)
            aucs.append(viz.average_precision)

            report = {}
            report['average_precision'] = viz.average_precision
            reports.append(report)


        means = operate_dict_const(operate_dict(reports, lambda a,b: a+b), lambda a,b:a/b, len(reports))
        squares = [operate_dict_const(report, lambda a,_:a**2, 0) for report in reports]
        sum_squares = operate_dict_const(operate_dict(squares, lambda a,b: a+b), lambda a,_: a/len(reports), 0)
        variance = operate_dict([sum_squares, operate_dict_const(means, lambda a,_: a**2, 0)], lambda a,b: a-b)
        std_error = operate_dict_const(variance, lambda a,_: np.sqrt(a)/np.sqrt(len(reports)), 0)
        scores[name] = {'means':means, 'std_error':std_error}

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[0] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            label=r"Mean PR of "+name+" (AUC = %0.3f $\pm$ %0.3f)" % (mean_auc, std_auc/np.sqrt(len(tprs))),
            lw=2,
            alpha=0.8,
        )

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr/np.sqrt(len(tprs)), 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr/np.sqrt(len(tprs)), 0)
        ax.fill_between(
            mean_fpr,
            tprs_lower,
            tprs_upper,
            color=color,
            alpha=0.2,
            label=r"$\pm$ 1 std. err. for "+name,
        )

        ax.set(
            xlim=[-0.05, 1.05],
            ylim=[-0.05, 1.05],
            title="Receiver operating characteristic example",
        )
        ax.legend(loc="lower right")



    plt.gcf()
    handles, labels = plt.gca().get_legend_handles_labels()
    l = [(i,name) for i,name in enumerate(labels) if not re.search("^ROC", name)]
    handles = [handles[i] for (i,_) in l]
    labels = [name for (_,name) in l]
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title(title)
    plt.savefig(path)
    plt.show()

    return scores