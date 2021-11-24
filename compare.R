library(data.table)

data <- fread('maggrad1D.csv')
data[,V2:=V2*2.82624272560609e-01]
model1 <- lm (V2~1, data)
summary(model1)

model2 <- lm (V2~V1, data)
summary(model2)

anova(model1, model2)
