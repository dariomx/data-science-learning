source("predict/test-data.r")

inc.cols = c("audience_score", "runtime", "critics_score",
             "best_pic_nom", "best_pic_win", "best_actor_win", 
             "best_actress_win", "best_dir_win"
             )
data = movies[(names(movies) %in% inc.cols)]
data$num_oscar = ifelse(data$best_pic_nom=="yes", 1, 0) + 
                 ifelse(data$best_pic_win=="yes", 1, 0) + 
                 ifelse(data$best_actor_win=="yes", 1, 0) + 
                 ifelse(data$best_actress_win=="yes", 1, 0) +
                 ifelse(data$best_dir_win=="yes", 1, 0)

lm.ret = lm(audience_score ~ runtime + critics_score + num_oscar, data=data)
summary(lm.ret)$adj.r.squared

# Correlation panel
panel.cor <- function(x, y){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- round(cor(x, y), digits=2)
  txt <- paste0("R = ", r)
  cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt, cex = cex.cor * r)
}
# Customize upper panel
upper.panel<-function(x, y){
  points(x,y, pch = 19)
}
# Create the plots
pairs(audience_score ~ runtime + critics_score + num_oscar, 
      data=data,
      lower.panel = panel.cor,
      upper.panel = upper.panel)


lm.ret = lm(audience_score ~ runtime + critics_score, data=data)
summary(lm.ret)$adj.r.squared

predict(lm.ret, test.data)