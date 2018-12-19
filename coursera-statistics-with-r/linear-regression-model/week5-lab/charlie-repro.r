library(ORE)

chaca.ore.connect <- function(...) 
  ore.connect(user="ore", password="welcome1", sid="chacadb", 
              host='slc06pqb', port=1521, all=TRUE)

chaca.ore.connect()

glm.ret = ore.odmGLM(audience_score ~ ., data=omovies)

sink("glm.ret")
print(glm.ret)
sink()

ore.disconnect()