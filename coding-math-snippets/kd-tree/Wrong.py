class KDTree:
    def __init__(self, x):
        self.x = x
        self.left = None
        self.right = None

k = 2

def insert(x, kdt, lev):
    if kdt is None:
        return KDTree(x, None, None)
    if x == kdt.x:
        return kdt # duplicate
    elif x[lev % k] < kdt.x[lev % k]:
        kdt.left = insert(x,
                          kdt.left,
                          lev + 1)
    else:
        kdt.right = insert(x,
                           kdt.right,
                           lev + 1)

