inc.cols = c("audience_score",
             "critics_score",
             "title_type",
             "runtime", 
             "genre", 
             "mpaa_rating", 
             "thtr_rel_month", 
             "thtr_rel_day",
             "dvd_rel_month", 
             "dvd_rel_day", 
             "best_pic_nom", 
             "best_pic_win", 
             "best_actor_win",  
             "best_actress_win", 
             "best_dir_win", 
             "top200_box")
data = movies[(names(movies) %in% inc.cols)]
lm.ret = lm(audience_score ~ ., data=data)
print(summary(lm.ret)$adj.r.squared)

# audience_score=84,
american.sniper = data.frame(critics_score=72,                             
                             title_type="Feature Film",
                             runtime=134,                             
                             genre="Drama",
                             mpaa_rating="R",
                             thtr_rel_month=01,
                             thtr_rel_day=16,
                             dvd_rel_month=05,
                             dvd_rel_day=19,
                             best_pic_nom="no",
                             best_pic_win="no",
                             best_actor_win="no",
                             best_actress_win="no",
                             best_dir_win="no",
                             top200_box="no")
predict(lm.ret, american.sniper)

american.sniper$genre = "Action & Adventure"
predict(lm.ret, american.sniper)
