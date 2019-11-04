# Python functions for QBUS2820
# Marcel Scharth, The University of Sydney Business School
# Yiran Jing
# August 2017

# Imports
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf



def rmse_jack(response, predicted):

    y = np.array((np.ravel(response)-np.ravel(predicted))**2)
    y_sum = np.sum(y)
    n = len(y)

    resample = np.sqrt((y_sum-y)/(n-1))

    rmse = np.sqrt(y_sum/n)
    se = np.sqrt((n-1)*np.var(resample))

    return rmse, se


def r2_jack(response, predicted):


    e2 = np.array((np.ravel(response)-np.ravel(predicted))**2)
    y2 = np.array((np.ravel(response)-np.mean(np.ravel(response)))**2)

    rss = np.sum(e2)
    tss = np.sum(y2)
    n = len(e2)

    resample = 1-(rss-e2)/(tss-y2)

    r2 = 1-rss/tss
    se = np.sqrt((n-1)*np.var(resample))

    return r2, se





## yiran define
def forwardselection_logisticl1(X, y):
    """Forward variable selection based on the Scikit learn API


    Output:
    ----------------------------------------------------------------------------------
    Scikit learn regression object for the best model
    """

    # Functions
    import sklearn.linear_model as skl_lm
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import cross_val_score

    # Initialisation
    base = []
    p = X.shape[1]
    candidates = list(np.arange(p))

    # Forward recursion
    i=1
    bestcvscore=-np.inf
    while i<=p:
        bestscore = 0
        for variable in candidates:
            model = skl_lm.LogisticRegressionCV(penalty='l1', solver='liblinear',class_weight='balanced')
            model.fit(X.iloc[:, base + [variable]], y)
            score = model.score(X.iloc[:, base + [variable]], y)
            if score > bestscore:
                bestscore = score
                best = model
                newvariable=variable
        base.append(newvariable)
        candidates.remove(newvariable)

        cvscore = cross_val_score(best, X.iloc[:, base], y, scoring='neg_log_loss').mean()

        if cvscore > bestcvscore:
            bestcvscore=cvscore
            bestcv = best
            subset = base[:]
        i+=1

    #Finalise
    return bestcv, subset


class forward_logisticl1:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model, self.subset = forwardselection_logisticl1(X, y)

    def predict(self, X):
        return self.model.predict(X.iloc[:, self.subset])

    def predict_proba(self, X):
        return self.model.predict_proba(X.iloc[:, self.subset])





## yiran define
def forwardselection_logisticl2(X, y):
    """Forward variable selection based on the Scikit learn API


    Output:
    ----------------------------------------------------------------------------------
    Scikit learn OLS regression object for the best model
    """

    # Functions
    import sklearn.linear_model as skl_lm
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import cross_val_score

    # Initialisation
    base = []
    p = X.shape[1]
    candidates = list(np.arange(p))

    # Forward recursion
    i=1
    bestcvscore=-np.inf
    while i<=p:
        bestscore = 0
        for variable in candidates:
            model = skl_lm.LogisticRegressionCV(penalty='l2')
            model.fit(X.iloc[:, base + [variable]], y)
            score = model.score(X.iloc[:, base + [variable]], y)
            if score > bestscore:
                bestscore = score
                best = model
                newvariable=variable
        base.append(newvariable)
        candidates.remove(newvariable)

        cvscore = cross_val_score(best, X.iloc[:, base], y, scoring='neg_log_loss').mean()

        if cvscore > bestcvscore:
            bestcvscore=cvscore
            bestcv = best
            subset = base[:]
        i+=1

    #Finalise
    return bestcv, subset


class forward_logisticl2:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model, self.subset = forwardselection_logisticl2(X, y)

    def predict(self, X):
        return self.model.predict(X.iloc[:, self.subset])

    def predict_proba(self, X):
        return self.model.predict_proba(X.iloc[:, self.subset])




## forward logistic
def forwardselection_logistic(X, y):
    """Forward variable selection based on the Scikit learn API


    Output:
    ----------------------------------------------------------------------------------
    Scikit learn OLS regression object for the best model
    """

    # Functions
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Initialisation
    base = []
    p = X.shape[1]
    candidates = list(np.arange(p))

    # Forward recursion
    i=1
    bestcvscore=-np.inf
    while i<=p:
        bestscore = 0
        for variable in candidates:
            model = LogisticRegression()
            model.fit(X.iloc[:, base + [variable]], y)
            score = model.score(X.iloc[:, base + [variable]], y)
            if score > bestscore:
                bestscore = score
                best = model
                newvariable=variable
        base.append(newvariable)
        candidates.remove(newvariable)

        cvscore = cross_val_score(best, X.iloc[:, base], y, scoring='neg_log_loss').mean()

        if cvscore > bestcvscore:
            bestcvscore=cvscore
            bestcv = best
            subset = base[:]
        i+=1

    #Finalise
    return bestcv, subset


class forward_logistic:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.model, self.subset = forwardselection_logistic(X, y)

    def predict(self, X):
        return self.model.predict(X.iloc[:, self.subset])

    def predict_proba(self, X):
        return self.model.predict_proba(X.iloc[:, self.subset])



## yiran define
class PCR_logistic:   ## with out CV
    def __init__(self, M=1):
        self.M=M

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        self.pca=PCA(n_components=self.M)
        Z= self.pca.fit_transform(X)
        self.pcr = LogisticRegression().fit(Z, y)

    def predict(self, X):
        return self.pcr.predict(self.pca.transform(X))

    def predict_proba(self, X):
        return self.pcr.predict_proba(self.pca.transform(X))

    def cv_score(self, X, y, cv=10):
        from sklearn.model_selection import cross_val_score
        Z=self.pca.transform(X)
        scores = cross_val_score(self.pcr, Z, np.ravel(y), cv=cv, scoring='neg_log_loss').mean()
        return np.sqrt(-1*np.mean(scores))


## yiran define
class PCR_CV_logistic:
    def __init__(self, M=1):
        self.M=M

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        self.pca=PCA(n_components=self.M)
        Z= self.pca.fit_transform(X)
        self.pcr = pcrCV_logistic(Z, y)

    def predict(self, X):
        return self.pcr.predict(self.pca.transform(X))

    def predict_proba(self, X):
        return self.pcr.predict_proba(self.pca.transform(X))

    def cv_score(self, X, y, cv=10):
        from sklearn.model_selection import cross_val_score
        Z=self.pca.transform(X)
        scores = cross_val_score(self.pcr, Z, np.ravel(y), cv=cv, scoring='neg_log_loss').mean()
        return np.sqrt(-1*np.mean(scores))

## yiran define
def pcrCV_logistic(X, y):
    # Approximate cross-validation
    from sklearn.model_selection import cross_val_score

    p=X.shape[1]
    bestscore= -np.inf
    cv_scores = []
    for m in range(1,p+1):
        model = PCR_logistic(M=m)  ## we use simple logistic for CV to select the best m
        model.fit(X, y)
        model.predict_proba(X)
        Z=model.pca.transform(X)
        score = cross_val_score(model.pcr, Z, y, cv=10, scoring='neg_log_loss').mean()
        cv_scores.append(score)
        if score > bestscore:
            bestscore=score
            best=model

    best.cv_scores = pd.Series(cv_scores, index = np.arange(1,p+1))
    return best



## yiran define(L1 shrinkage PCR)
class PCR_logistic_l1:   ## with out CV
    def __init__(self, M=1):
        self.M=M

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        self.pca=PCA(n_components=self.M)
        Z= self.pca.fit_transform(X)
        self.pcr = LogisticRegression(penalty='l1', solver='liblinear').fit(Z, y)

    def predict(self, X):
        return self.pcr.predict(self.pca.transform(X))

    def predict_proba(self, X):
        return self.pcr.predict_proba(self.pca.transform(X))

    def cv_score(self, X, y, cv=10):
        from sklearn.model_selection import cross_val_score
        Z=self.pca.transform(X)
        scores = cross_val_score(self.pcr, Z, np.ravel(y), cv=cv, scoring='neg_log_loss').mean()
        return np.sqrt(-1*np.mean(scores))


## yiran define  + l1
class PCR_CV_logistic_l1:
    def __init__(self, M=1):
        self.M=M

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        self.pca=PCA(n_components=self.M)
        Z= self.pca.fit_transform(X)
        self.pcr = pcrCV_logistic_l1(Z, y)

    def predict(self, X):
        return self.pcr.predict(self.pca.transform(X))

    def predict_proba(self, X):
        return self.pcr.predict_proba(self.pca.transform(X))

    def cv_score(self, X, y, cv=10):
        from sklearn.model_selection import cross_val_score
        Z=self.pca.transform(X)
        scores = cross_val_score(self.pcr, Z, np.ravel(y), cv=cv, scoring='neg_log_loss').mean()
        return np.sqrt(-1*np.mean(scores))






## yiran define  + l1
def pcrCV_logistic_l1(X, y):
    # Approximate cross-validation
    from sklearn.model_selection import cross_val_score

    p=X.shape[1]
    bestscore= -np.inf
    cv_scores = []
    for m in range(1,p+1):
        model = PCR_logistic_l1(M=m)  ## we use simple logistic for CV to select the best m
        model.fit(X, y)
        model.predict_proba(X)
        Z=model.pca.transform(X)
        score = cross_val_score(model.pcr, Z, y, cv=10, scoring='neg_log_loss').mean()
        cv_scores.append(score)
        if score > bestscore:
            bestscore=score
            best=model

    best.cv_scores = pd.Series(cv_scores, index = np.arange(1,p+1))
    return best








## yiran define +l2
class PCR_logistic_l2:   ## with out CV
    def __init__(self, M=1):
        self.M=M

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        self.pca=PCA(n_components=self.M)
        Z= self.pca.fit_transform(X)
        self.pcr = LogisticRegression(penalty='l2').fit(Z, y)

    def predict(self, X):
        return self.pcr.predict(self.pca.transform(X))

    def predict_proba(self, X):
        return self.pcr.predict_proba(self.pca.transform(X))

    def cv_score(self, X, y, cv=10):
        from sklearn.model_selection import cross_val_score
        Z=self.pca.transform(X)
        scores = cross_val_score(self.pcr, Z, np.ravel(y), cv=cv, scoring='neg_log_loss').mean()
        return np.sqrt(-1*np.mean(scores))


## yiran define  + l2
class PCR_CV_logistic_l2:
    def __init__(self, M=1):
        self.M=M

    def fit(self, X, y):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression

        self.pca=PCA(n_components=self.M)
        Z= self.pca.fit_transform(X)
        self.pcr = pcrCV_logistic_l2(Z, y)

    def predict(self, X):
        return self.pcr.predict(self.pca.transform(X))

    def predict_proba(self, X):
        return self.pcr.predict_proba(self.pca.transform(X))

    def cv_score(self, X, y, cv=10):
        from sklearn.model_selection import cross_val_score
        Z=self.pca.transform(X)
        scores = cross_val_score(self.pcr, Z, np.ravel(y), cv=cv, scoring='neg_log_loss').mean()
        return np.sqrt(-1*np.mean(scores))






## yiran define +l2
def pcrCV_logistic_l2(X, y):
    # Approximate cross-validation
    from sklearn.model_selection import cross_val_score

    p=X.shape[1]
    bestscore= -np.inf
    cv_scores = []
    for m in range(1,p+1):
        model = PCR_logistic_l2(M=m)  ## we use simple logistic for CV to select the best m
        model.fit(X, y)
        model.predict_proba(X)
        Z=model.pca.transform(X)
        score = cross_val_score(model.pcr, Z, y, cv=10, scoring='neg_log_loss').mean()
        cv_scores.append(score)
        if score > bestscore:
            bestscore=score
            best=model

    best.cv_scores = pd.Series(cv_scores, index = np.arange(1,p+1))
    return best






## yiran define :
def linear_l1_svc_cv(X, y):

    from sklearn.svm import LinearSVC
    from sklearn.model_selection import cross_val_score

    p=X.shape[1]
    bestscore=-np.inf
    for m in range(1,p): # not fitting with M=p avoids occasional problems
        pls = PLSRegression(n_components=m).fit(X, y)
        score = cross_val_score(pls, X, y, cv=10, scoring='neg_log_loss').mean()
        if score > bestscore:
            bestscore=score
            best=pls
    return best





def plsCV_logistic(X, y):

    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import cross_val_score

    p=X.shape[1]
    bestscore=-np.inf
    for m in range(1,p): # not fitting with M=p avoids occasional problems
        pls = PLSRegression(n_components=m).fit(X, y)
        score = cross_val_score(pls, X, y, cv=10, scoring='neg_log_loss').mean()
        if score > bestscore:
            bestscore=score
            best=pls
    return best
