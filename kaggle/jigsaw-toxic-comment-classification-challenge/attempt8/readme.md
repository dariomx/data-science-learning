0. Get vocabulary only from toxic comments (n-grams 1-5)

0.5 Prune vocabulary based on tf-idf

1. Linguistically normalize(pre-process) comments as bag of words

2. Bag of words into counter vectors (save as scipy sparse matrices)

3. Train and test with sklearn SVD

