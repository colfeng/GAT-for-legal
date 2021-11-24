import numpy as np

def CAIL2018_S(r, p):
    one = np.ones(len(r))
    r_log = np.log(list(np.array(r) + one))
    p_log = np.log(list(np.array(p) + one))
    V = list(map(lambda x: x[0] - x[1], zip(r_log, p_log)))
    V = list(map(abs, V))
    S = np.zeros((len(r), 1))
    for i, v in enumerate(V):
        if v <= 0.2:
            S[i] = 1
        elif 0.2 < v <= 0.4:
            S[i] = 0.8
        elif 0.4 < v <= 0.6:
            S[i] = 0.6
        elif 0.6 < v <= 0.8:
            S[i] = 0.4
        elif 0.8 < v <= 1:
            S[i] = 0.2
        elif v > 1:
            S[i] = 0
    return S

def Exact_Match(r, p):
    V = list(map(lambda x: x[0] - x[1], zip(r, p)))
    count = 0
    for v in V:
        if -1 < v < 1:
            count += 1
    EM = count / len(r)
    return EM

def Acc(r, p, k):
    count = 0
    V = list(map(lambda x: x[0] - x[1], zip(r, p)))
    np.seterr(divide='ignore', invalid='ignore')
    V = [abs(v) / r[i] for i, v in enumerate(V)]
    for v in V:
        if v <= k:
            count += 1
    Acc = count / len(r)
    return Acc