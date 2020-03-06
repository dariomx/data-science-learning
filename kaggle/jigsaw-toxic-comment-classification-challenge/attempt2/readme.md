same as previous attempt, except that we use nltk
implementation of Naïve Bayes classifier

we also restrict the features to the words from toxic comments
(meaning, those which have at least one category flagged). this was
needed due the huge size of vocabulary; per technique shown here
(see Document Classification section):

nltk.org/book/ch06.html

used 2k words as the example, but program never ended. need a more
refined vocabulary I think.