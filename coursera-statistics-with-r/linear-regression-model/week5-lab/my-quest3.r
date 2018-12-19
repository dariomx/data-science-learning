source("predict/test-data.r")

inc.cols = c("audience_score", "critics_score")
#inc.cols = c(inc.cols, c("runtime"))
#inc.cols = c(inc.cols, c("genre", "title_type"))
data = movies[(names(movies) %in% inc.cols)]

summary(lm(audience_score ~ ., data=data))

predict(lm.ret, test.data)
