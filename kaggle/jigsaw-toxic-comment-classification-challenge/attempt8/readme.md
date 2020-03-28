0. Linguistically normalize(pre-process) comments as bag of words

1. Get vocabulary only from toxic comments (n-grams 1)

1.5 Prune vocabulary (pick those only used in toxic comments)

2. Bag of words into counter vectors (save as scipy sparse matrices)

3. Train and test with sklearn SVM

