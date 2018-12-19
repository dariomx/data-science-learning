score.model = function(inc.cols) {
  data = movies[(names(movies) %in% inc.cols)]
  lm.ret = lm(audience_score ~ ., data=data)
  return (summary(lm.ret)$adj.r.squared)
}

inc.cols = c("audience_score", 
             "runtime", "genre", "mpaa_rating", 
             "critics_score", "best_pic_nom", "top200_box",
             "studio", "director", 
             "dvd_rel_year", "thtr_rel_month")
score.model(inc.cols)

# 2018-11-21 00:18:48: audience_score runtime mpaa_rating studio 
# thtr_rel_month dvd_rel_year critics_score top200_box director: 0.704928
inc.cols = c("audience_score", 
             "runtime", "mpaa_rating", 
             "critics_score", "top200_box",
             "studio", "director", 
             "dvd_rel_year", "thtr_rel_month")
score.model(inc.cols)

inc.cols = c("audience_score", 
             "runtime", "mpaa_rating", 
             "critics_score", "top200_box",
             "studio", "director", 
             "thtr_rel_month")
score.model(inc.cols)

# genre mpaa_rating studio critics_score best_pic_nom top200_box director
inc.cols = c("audience_score", 
             "genre", "mpaa_rating", 
             "critics_score", "best_pic_nom", "top200_box",
             "studio", "director")
score.model(inc.cols)

inc.cols = c("audience_score", 
             "runtime", "genre", "mpaa_rating", 
             "critics_score", "best_pic_nom", "top200_box",
             "studio", "director")
score.model(inc.cols)

inc.cols = c("audience_score", 
             "runtime", "genre", "mpaa_rating", "thtr_rel_month",
             "critics_score", "best_pic_nom", "top200_box")
score.model(inc.cols)

