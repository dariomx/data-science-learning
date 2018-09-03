from random import randint

def sample(A, k):
    B = A[:]
    ans = []
    for _ in range(k):
        j = randint(0, len(B)-1)
        ans.append(B.pop(j))
    return ans

arr = list(range(100))
k = 10
print(sample(arr, k))



