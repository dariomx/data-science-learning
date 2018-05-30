from random import randint

def sample(A, k):
    R = A[:k]
    n = len(A)
    for i in range(k, n):
        j = randint(0, i)
        if j < k:
            R[j] = A[i]
    return R

