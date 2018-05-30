from random import randint

def sample(A, k):
    n = len(A)
    used = set()
    ans = []
    while len(ans) < k:
        j = randint(0, n-1)
        if j in used:
            continue
        used.add(j)
        ans.append(A[j])
    return ans

arr = list(range(100))
k = 10
print(sample(arr, k))
