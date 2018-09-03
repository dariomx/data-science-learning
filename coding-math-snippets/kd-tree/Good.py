class KDTree:
    def __init__(self, x, left, right):
        self.x = x
        self.left = left
        self.right = right

k = 2


def median(xs, param):
    pass


def build(xs, lev):
    if xs is None:
        return None
    i = lev % k
    med = median(xs, i)
    left = [x for x in xs if x[i] < med[i]]
    right = [x for x in xs if x[i] > med[i]]
    return KDTree(xs[i],
                  build(left, lev + 1),
                  build(right, lev + 1))


