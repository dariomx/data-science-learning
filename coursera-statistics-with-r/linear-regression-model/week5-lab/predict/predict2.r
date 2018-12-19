inc.cols = c("audience_score", 
             "runtime", "genre", "mpaa_rating", "thtr_rel_month",
             "critics_score", "best_pic_nom", "top200_box")
data = movies[(names(movies) %in% inc.cols)]
lm.ret = lm(audience_score ~ ., data=data)
print(summary(lm.ret)$adj.r.squared)

# audience_score=84,
american.sniper = data.frame(runtime=134,
                             genre="Drama",
                             mpaa_rating="R",
                             thtr_rel_month=01,
                             critics_score=72,
                             best_pic_nom="no",
                             top200_box="no",
                             studio="Warner Bros. Pictures",
                             director="Clint Eastwood")
predict(lm.ret, american.sniper)

american.sniper$genre = "Action & Adventure"
predict(lm.ret, american.sniper)