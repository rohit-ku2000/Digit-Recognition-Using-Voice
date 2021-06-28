import numpy as np
# util functions
def MA(x, N):
    return np.convolve(x, np.ones((N,)) / N)[(N - 1):]


def filt_amp(x):
    r = []
    for i in x:
        if i < np.max(x) / 10:
            r.append(0)
        else:
            r.append(i)
    return (np.asarray(r))


def bound(y):
    x = y
    i1 = np.argmax(x)
    bound = []
    l = 0
    r = 1
    while x[i1 - l] > 0:
        x[i1 - l] = 0
        l = l + 1
    while x[i1 + r] > 0:
        x[i1 + r] = 0
        r = r + 1

    bound.append(i1 - l)
    bound.append(i1 + r)

    i2 = np.argmax(x)
    l = 0
    r = 1
    while x[i2 - l] > 0:
        x[i2 - l] = 0
        l = l + 1
    while x[i2 + r] > 0:
        x[i2 + r] = 0
        r = r + 1
    bound.append(i2 - l)
    bound.append(i2 + r)
    bound = np.sort(np.asarray(bound))
    return (bound)