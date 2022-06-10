

from scoping import scoping
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, OPTICS
import sfztest_1agent_acc as model
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import collections
from sklearn.metrics import silhouette_score, mean_squared_error
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import davies_bouldin_score, r2_score
import warnings


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def constant(x, a):
    return a


def linear(x, a, b):
    return np.poly1d([a, b])(x)


def quadratic(x, a, b, c):
    return np.poly1d([a, b, c])(x)


def cubic(x, a, b, c, d):
    return np.poly1d([a, b, c, d])(x)


def siglinear(x, a, b):
    return sigmoid(np.poly1d([a, b])(x))


def sigquadratic(x, a, b, c):
    return sigmoid(np.poly1d([a, b, c])(x))


def sigcubic(x, a, b, c, d):
    return sigmoid(np.poly1d([a, b, c, d])(x))


def siglinear_sin(x, a, b, c):
    return sigmoid(np.poly1d([a, b])(x)) + np.sin(c)


def sin(x, a, b, c, d):
    return a * np.sin(b * x+c) + d


fs = [linear, quadratic, cubic, siglinear, sigquadratic, sigcubic, sin]
fsac = {linear: 2, quadratic: 3, cubic: 4,
        siglinear: 2, sigquadratic: 3, sigcubic: 4, sin: 4}

X1 = [0.79, .85, .9, 1, 1.1, 1.29]
Y1 = [0.9, .85, .85, 0.5, 0.5, 0.9]
X2 = [0.79, .85, .9, 1, 1.1, 1.29]
Y2 = [0.5, .5, .65, 0.8, 0.5, 0.4]
X3 = [0.79, 0.81, .85, .9, 1, 1.1, 1.29]
Y3 = [0.6, .5, .7, 0.8, 0.2, 0.4, .5]
X4 = [0.5, 0.7, 1, 1.3, 1.5]
Y4 = [0.5, 0.3, 0, 0.3, 0.5]
X5 = [0.5, 0.7, 1, 1.3, 1.5]
Y5 = [0.9, 0.3, 0.8, 0.2, 0.7]
X6 = [0.5, 0.7, 1, 1.3, 1.5]
Y6 = [0.9, 0.6, .1, 0.1, 0.1]
X7 = [0.5, 0.7, 1, 1.3, 1.5]
Y7 = np.poly1d([1, 0, 0])(X7)
Y7 = np.where(Y7 <= 1, Y7, 0.9)
X8 = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
Y8 = [0.1, 1, 0.5, 0.9, 0.1, 1]
X9 = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
Y9 = [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]


def Draw(x, y):
    stds = []
    try:
        for f in fs:
            coeffient, _ = curve_fit(
                f, x, y)
            yline = f(x, *coeffient)
            rsquare = r2_score(y, yline)
            adjrsqare = round(1-(1-rsquare)*(x.size-1) /
                              (x.size-fsac[f]-1+0.0001), 3)
            stds.append((adjrsqare, f, coeffient))
    except:
        pass
    r = np.array(model.arange(x.min(), x.max(), 0.01))
    func = stds[np.argmin([x[0] for x in stds])][1]
    coeffient = stds[np.argmin([x[0] for x in stds])][2]
    plt.plot(r, func(r, *coeffient))
    plt.scatter(x, y)
    plt.show()


def CreateEstimate(x, y, size, itl, verbose):
    x, y = np.array(x), np.array(y)
    data = collections.defaultdict(list)
    np.random.seed(0)
    spl = UnivariateSpline(x, y, s=0)
    r = model.arange(x.min(), x.max(), itl)
    for _ in range(size):
        an = np.random.choice(r)
        pi = spl(an)
        pi = min(1, pi)
        pi = max(0, pi)
        data[an].append(np.random.binomial(1, pi))
    est = {x: np.mean(data[x]) for x in data}
    est = dict(sorted(est.items()))
    Xest = np.array(list(est.keys()))
    Yest = np.array(list(est.values()))

    # if verbose:
    #     _, ax = plt.subplots(2)
    #     r = model.arange(x.min(), x.max(), 0.01)
    #     yline = spl(r)
    #     yline = np.where(yline <= 1, yline, 1)
    #     ax[0].plot(r, yline, x, y, 'o')
    #     ax[1].plot(Xest, Yest, "-go", x, y, "ro")
    #     plt.draw()
    return Xest, Yest, spl


def Fitting(X, Y, ax):
    X, Y = np.array(X), np.array(Y)
    X_copy, Y_copy = X.copy(), Y.copy()
    norm = np.stack((X, Y), axis=1)

    def GetBestK(method=0):
        labels = []
        silhouette_avg = []

        def LabelTogether(seg, label):
            if seg == 1:
                return False
            for i in range(seg):
                j = np.where(label == i)[0]
                if not j[-1]-j[0]+1 == np.count_nonzero(label == i):
                    return False
            return True

        for i in range(2, 6):
            labellist = [s for s in [SpectralClustering(n_clusters=i).fit_predict(norm), AgglomerativeClustering(n_clusters=i).fit_predict(norm),
                                     ] if LabelTogether(i, s)]
            listscore = [davies_bouldin_score(
                norm, label)*((i-1)/i)**2 for label in labellist]
            # print(i, listscore, labellist)
            # KMeans(n_clusters=i).fit_predict(norm),SpectralClustering(n_clusters = i).fit_predict(norm),
            # AgglomerativeClustering(n_clusters=i).fit_predict(norm)
            # metrics.calinski_harabasz_score(norm, label), davies_bouldin_score, silhouette_score
            # if labellist:
            #     mid = np.argmin(listscore)
            #     silhouette_avg.append((i-2, listscore[mid]))
            #     labels.append(labellist[mid])
            for j, _ in enumerate(labellist):
                silhouette_avg.append((i-2, listscore[j]))
                labels.append(labellist[j])

        if not silhouette_avg:
            def GroupNegative(a):
                k = np.where(a == -1)[0]
                start = -2
                ind = a.max()
                for l in k:
                    if l != start+1:
                        ind += 1
                    start = l
                    a[l] = ind
                return a
            labellist = [k for i in range(2, 6)
                         if LabelTogether((k := GroupNegative(OPTICS(min_samples=i).fit_predict(norm))).max()+1, k)]

            listscore = [davies_bouldin_score(
                norm, label) for label in labellist]
            for j, s in enumerate(labellist):
                silhouette_avg.append((s.max()-1, listscore[j]))
                labels.append(labellist[j])
        if not silhouette_avg:
            silhouette_avg.append((X.size-2, 0))
            labels.append([*range(X.size)])
        return silhouette_avg[(k := np.argmin([x[1] for x in silhouette_avg]))][0], labels[k]
    best_k, best_label = GetBestK()
    best_segment = best_k+2
    print(best_label)

    def get_interval(segment, label):
        label = np.array(label)
        idx = 0
        rg = [[None, np.NINF]]
        for _ in range(segment):
            itv = (label == label[idx]).nonzero()[0]
            rg.append([rg[-1][1], X[itv[-1]]])
            idx = itv[-1]+1
        rg[-1][1] = np.inf
        rg.pop(0)
        return rg

    segment = get_interval(best_segment, best_label)
    print(segment)
    for i in range(len(segment)-1):
        X = np.concatenate((X, [segment[i][1]+0.01]))
        Y = np.concatenate((Y, Y[np.where(X == segment[i][1])]))
    zxy = sorted(list(zip(X, Y)))
    X, Y = np.array([x[0] for x in zxy]), np.array([x[1] for x in zxy])

    def BestCurveWithoutOutliers(X, Y):
        warnings.filterwarnings("ignore")

        def GetBestCurve(x, y):
            if x.size == 1:
                return constant, [y[0]]
            stds = []
            for f in fs:
                try:
                    if x.size < fsac[f]:
                        continue
                    coeffient, _ = curve_fit(
                        f, x, y)
                    yline = f(x, *coeffient)
                    rsquare = r2_score(y, yline)
                    rsquare = max(0, rsquare)
                    if x.size > 5:
                        adjrsqare = round(1-(1-rsquare)*(x.size-1) /
                                            (x.size-fsac[f]-1), 3)
                    else:
                        adjrsqare = rsquare
                    stds.append((adjrsqare, f, coeffient))
                except RuntimeError:
                    pass
            return stds[(mid := np.argmax([x[0] for x in stds]))][1], stds[mid][2]

        sunit = []
        for lb, ub in segment:
            s = (X > lb)*(X <= ub)
            sunit.append((X[s], Y[s]))
        bf = []
        for x, y in sunit:
            bf.append(GetBestCurve(x, y))
            print(bf[-1][0])
        outsig = np.full((X.size), True)

        outsig = np.array([], dtype=bool)
        for i, (x, y) in enumerate(sunit):
            f, coe = bf[i]
            outliers = np.full((x.size), True)
            if not (x.size == 1 or x.size == fsac[f]):
                residual = np.absolute(
                    np.array([y[j]-f(x[j], *coe) for j in range(len(y))]))
                sor = np.sqrt(
                    np.sum(np.square(residual))/(x.size-fsac[f]))
                outliers = np.where(residual > 2 * sor, False, True)
                outliers[0] = outliers[-1] = True
            outsig = np.concatenate((outsig, outliers))

        #     r = np.array(model.arange(min(x), max(x), 0.01))
        #     plt.plot(r, f(r, *coe))
        #     plt.scatter(x[outliers], y[outliers], color="blue")
        #     plt.scatter(x[~outliers], y[~outliers], color="red")
        # plt.show()

        X = X[outsig]
        Y = Y[outsig]
        if np.all(outsig):
            return bf, (X, Y)
        else:
            return BestCurveWithoutOutliers(X, Y)
    bf, (X, Y) = BestCurveWithoutOutliers(X, Y)

    def Fit(x):
        v = 0
        for i in range(len(segment)):
            f, coe = bf[i]
            v += f(x, *coe) * np.multiply(x >
                                          segment[i][0], x <= segment[i][1])

        return v

    full_range = np.array(model.arange(X.min(), X.max(), 0.01))
    spl = UnivariateSpline(X_copy, Fit(X_copy), s=0, k=2)
    ax.plot(full_range, spl(full_range))
    for i in range(best_segment):
        filter = (X_copy > segment[i][0]) * (X_copy <= segment[i][1])
        ax.scatter(X_copy[filter], Y_copy[filter])
    return spl


xset = [X1, X2, X3, X4, X5, X6, X7, X8, X9]
yset = [Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9]


def ShowAllTest():
    fig, ax = plt.subplots(2, 4)
    fig.set_size_inches(1920/100, 900/100)
    fig.tight_layout()

    for i, (a, b) in enumerate([[x0, y0] for x0 in [0, 1] for y0 in [0, 1, 2, 3]]):
        with scoping():
            ax = ax[a][b]
            setid = i
            X = xset[setid]
            Y = yset[setid]
            itl = 0.1
            size = len(model.arange(min(X), max(X), itl)) * 30
            x3est, y3est, spl = CreateEstimate(X, Y, size, itl, 0)
            f = Fitting(x3est, y3est, ax)
            r = np.array(model.arange(min(X), max(X), 0.01))
            yline = spl(r)
            yline = np.where(yline <= 1, yline, 1)
            ax.plot(r, yline, "r")
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right-x_left)/(y_low-y_high)*.8))
    plt.show()


# ShowAllTest()
