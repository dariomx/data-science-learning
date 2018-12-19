source("predict/test-data.r")

inc.cols = c("audience_score", 
#             "critics_score", 
             "genre", 
             "best_pic_nom",
             "imdb_rating")
data = movies[,(names(movies) %in% inc.cols)]

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
pairs(audience_score ~ ., 
      data=data,
      lower.panel = panel.cor,
      upper.panel = upper.panel)


lm.ret = lm(audience_score ~ ., data=data)
summary(lm.ret)
predict(lm.ret, test.data)

