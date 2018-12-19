library(ORE)

chaca.ore.connect <- function(...) 
  ore.connect(user="ore", password="welcome1", sid="chacadb", 
              host='slc06pqb', port=1521, all=TRUE)

chaca.ore.connect()

print(ore.odmAI(audience_score ~ ., data=omovies, auto.data.prep=TRUE))

print(ore.odmAI(audience_score ~ ., data=omovies, auto.data.prep=FALSE))

summary(ore.stepwise(audience_score ~ ., data=omovies_act, direction="none"))

summary(ore.stepwise(audience_score ~ ., data=omovies, direction="none"))

ore.disconnect()