0. Linguistically normalize(pre-process) comments as bag of words

1. Get vocabulary only from toxic comments (n-grams 1-5, limit size
decreasingly with the n)

1.5 Prune vocabulary (pick top half from toxic-only words, and top half from
normal-only words)

2. Bag of words into one-hot vectors (save as scipy sparse matrices), but invent
a new category (non toxic comments)

3. Train and test with sklearn SVM


