library(ggplot2); library(caret)
install.packages("rpart")
library(rpart): library
library(randomForest)
pml.training <- read.csv("~/Desktop/Machine Learning/MachineLearningProjectWriteup/pml-training.csv")
pml.testing <- read.csv("~/Desktop/Machine Learning/MachineLearningProjectWriteup/pml-testing.csv")
trainingset<-pml.training[,colSums(is.na(pml.training)) == 0]
validationset <-pml.testing[,colSums(is.na(pml.testing)) == 0]
##summary(trainingset)

library(rattle)
library(rpart)

nzv <- nearZeroVar(trainingset, saveMetrics= TRUE)
##nzv[nzv$nzv,][1:10,]

dim(trainingset)

nzv <- nearZeroVar(trainingset)
filteredTrain <- trainingset[, -nzv]
filteredTrain <- filteredTrain[, -c(1:7)]
dim(filteredTrain)

nzv1 <- nearZeroVar(validationset)
filteredVal <- validationset[, -nzv1]
filteredVal <- filteredVal[, -c(1:7)]

inTrain <- createDataPartition(y=filteredTrain$classe, p=0.6, list=FALSE)
train <- filteredTrain[inTrain,]
test <- filteredTrain[-inTrain,]

set.seed(1234)

modelFitClass <- rpart(classe ~ ., data=train, method="class")
modelFitClass
predictionClass <- predict(modelFitClass, test, type = "class")
rpart.plot(modelFitClass, main="Classification Tree", extra=102, under=TRUE, faclen=0)
confusionMatrix(predictionClass, test$classe)

modFit <- train(classe~.,data=train,method="rf",trControl=trainControl(method="cv",number=5),prox=TRUE, allowParallel=TRUE)
modFit
print(modFit$finalModel)
getTree(modFit$finalModel,k=2)

predictions <- predict(modFit, newdata=test)
print(confusionMatrix(predictions, test$classe), digits=4)

modFitA1 <- rpart(classe ~ ., data=train, method="class")
fancyRpartPlot(modFit)

modFitBag <- train(classe~.,data=train,method="gbm",verbose=FALSE)
print(modFitBag)

qplot(predict(modFitBag,test),classe,data=test)

predictions1 <- predict(modFitBag, newdata=test)
print(confusionMatrix(predictions1, test$classe), digits=4)

modFit2 <- randomForest(classe ~. , data=train, method="class",keep.forest=TRUE)
modFit2
predictions2 <- predict(modFit2, newdata=test, type="class")
print(confusionMatrix(predictions2, test$classe), digits=4)
getTree(modFit2$finalModel,k=2)
##prediction2 <- predict(model2, subTesting, type = "class")