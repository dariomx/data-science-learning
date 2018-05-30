from random import shuffle

def sample(A, k):
    B = A[:]
    shuffle(B)
    return B[:k]

arr = list(range(100))
k = 10
print(sample(arr, k))
