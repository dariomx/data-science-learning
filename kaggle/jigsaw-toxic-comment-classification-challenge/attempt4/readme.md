same as previous attempt but extending to n-grams
(not sure which size, mmm, 1 to 5?)

we also ensure that each n-size for n-grams gets
a decent amount of top candidates.

we are excluding single words (1-grams), cause is hard to
distinguish bad words from the top-k portion; common words
get mixed ... well that was attempt a), but in b) we tried
including them as well along with bigger samples overall:

MAX_VOCAB = {1: 250, 2: 200, 3: 150, 4: 100, 5: 50}

and given higher rates of b), we went even further with c):

in d), we expand the vocab to all comments, not just the toxic ones

