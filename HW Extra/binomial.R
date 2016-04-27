library(ggplot2) 
library(MASS)
library(glmnet)
library(gdata)
X = read.table('data/matrix.txt', header = FALSE)
X = t(X)
Y = read.table('data/tumor.txt', header = FALSE)
Y[Y<=0] = 0
Y[Y>0] = 1
n = nrow(X)
split = 0.65
train = sample(1:n, round(n * split))
Xtrain = X[train, ]
Xtest = X[-train, ]
ytrain = Y[train,]
ytest = Y[-train,]
def_lasso_model <- cv.glmnet(Xtrain, ytrain, alpha =1.0, type.measure = "class", nfolds = 10, family='binomial') 
bestlambdalasso = def_lasso_model$lambda.min #0.0003839358
error_lasso = sum(def_lasso_model$cvm)/length(def_lasso_model$cvm) #0.8810
jpeg('def_lasso_model_graph_auc.jpeg')
# coef(def_lasso_model, s = "lambda.min")
plot(def_lasso_model)
dev.off()
# AUC = 0.7053863
# Deviance = 1.3353
# predicted = predict(def_lasso_model, as.matrix(Xtest), s = bestlambdalasso)
# predicted[predicted  < 0.5] = 0
# predicted[predicted  >= 0.5] = 1
# mean(predicted==ytest)

