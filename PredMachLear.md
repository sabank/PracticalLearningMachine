# Prediction Machine Learning
Sabank  
December 27, 2015  

#### Synopsis:
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.

The following analysis uses a random forest-based prediction algorithm to accomplish this task. This type of algorithm is usually suited for providing categorical outputs. With an estimated out-of-sample error of 0.14%, the model achieves an accuracy of 99.86% in predicting the quality of execution of a particular activity.

#### 1. Introduction
Six participants were asked to perform one set of 10 repetitions of barbell lifts in five different fashions, categorized as classes A, B, C, D and E. Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.

In this project, our goal was to use data from accelerometers on the bell, forearm, arm, and dumbell of the participants to build a prediction algorithm that takes these sensor readings and, given both a training and test dataset, correctly predicts the corresponding class (A to E).

More information on the whole dataset is available here: Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/har). Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13). Stuttgart, Germany: ACM SIGCHI, 2013.

#### 2. Data Preparation
##### 2.1 Setting analysis environment

```r
set.seed(987)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(e1071)
```

##### 2.2 Importing data

```r
# note: change 'https' for 'http' for calling 'read.csv(url())'
train_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# initial inspection of files allows to set read.csv command as follow
ptrain <- read.csv(url(train_url),header=TRUE,sep =",",na.strings=c("NA","#DIV/0!",""))
ptest <- read.csv(url(test_url),header=TRUE,sep =",",na.strings=c("NA","#DIV/0!",""))

dim(ptrain)
```

```
## [1] 19622   160
```

##### 2.3 Processing data

```r
# considering dimensions of training set, let's subset training set to create a validation set for estimating the out-of-sample error, and let's partition this set with p=3/4
inTrain <- createDataPartition(y=ptrain$classe,p=0.75,list=FALSE)
ptrain_train <- ptrain[inTrain,]
ptrain_valid <- ptrain[-inTrain,]

# exploratory analysis allows to select relevant variables as follow
# remove variables that are not sensor readings
ptrain_train <- ptrain_train[,-(1:5)]
ptrain_valid <- ptrain_valid[,-(1:5)]

# remove variables with nearly zero variance
nzv <- nearZeroVar(ptrain_train)
ptrain_train <- ptrain_train[,-nzv]
ptrain_valid <- ptrain_valid[,-nzv]

# remove variables that are almost always NA
allNA <- sapply(ptrain_train,function(x) mean(is.na(x))) > 0.95
ptrain_train <- ptrain_train[,allNA==FALSE]
ptrain_valid <- ptrain_valid[,allNA==FALSE]
```

#### 3. Predictive Model Building
##### 3.1 Training the Predictor

```r
# use 3-fold cross-validation
fitControl <- trainControl(method="cv",number=3,verboseIter=FALSE)

# fit model on ptrain_train and print final model parameters
fit <- train(classe ~ .,data=ptrain_train,method="rf",trControl=fitControl)
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.26%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4182    2    0    0    1 0.0007168459
## B    9 2834    3    2    0 0.0049157303
## C    0    8 2559    0    0 0.0031164784
## D    0    0   11 2401    0 0.0045605307
## E    0    0    0    3 2703 0.0011086475
```

##### 3.2 Evaluation

```r
# use model to predict 'classe' in ptrain_valid and print confusion matrix
evalpred <- predict(fit,newdata=ptrain_valid)
confusionMatrix(ptrain_valid$classe,evalpred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    1  948    0    0    0
##          C    0    1  854    0    0
##          D    0    0    1  802    1
##          E    0    0    0    3  898
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9986          
##                  95% CI : (0.9971, 0.9994)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9982          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9989   0.9988   0.9963   0.9989
## Specificity            1.0000   0.9997   0.9998   0.9995   0.9993
## Pos Pred Value         1.0000   0.9989   0.9988   0.9975   0.9967
## Neg Pred Value         0.9997   0.9997   0.9998   0.9993   0.9998
## Prevalence             0.2847   0.1935   0.1743   0.1642   0.1833
## Detection Rate         0.2845   0.1933   0.1741   0.1635   0.1831
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9996   0.9993   0.9993   0.9979   0.9991
```

##### 3.3 Retraining the Model

```r
# remove variables that are not sensor readings
ptrain <- ptrain[,-(1:5)]
ptest <- ptest[,-(1:5)]

# remove variables with nearly zero variance
nzv <- nearZeroVar(ptrain)
ptrain <- ptrain[,-nzv]
ptest <- ptest[,-nzv]

# remove variables that are almost always NA
allNA <- sapply(ptrain,function(x) mean(is.na(x))) > 0.95
ptrain <- ptrain[,allNA==FALSE]
ptest <- ptest[,allNA==FALSE]

# re-fit model on ptrain and print final model parameters
fitControl <- trainControl(method="cv",number=3,verboseIter=FALSE)
fit <- train(classe ~ .,data=ptrain,method="rf",trControl=fitControl)
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.12%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5578    1    0    0    1 0.0003584229
## B    4 3791    2    0    0 0.0015801949
## C    0    4 3418    0    0 0.0011689071
## D    0    0    9 3206    1 0.0031094527
## E    0    0    0    2 3605 0.0005544774
```

#### 4. Application to 'Test' Set
##### 4.1 Predictions

```r
answers <- predict(fit,newdata=ptest)
answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

##### 4.2 File creation

```r
# convert predictions to character vector
answers <- as.character(answers)
# create function to write predictions to files
pml_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}
# create prediction files to submit
pml_write_files(answers)
```

#### 5. Conclusion
The random forest-based model applied to the test set provides an accuracy of 99.86% with 3-fold cross-validation parameter, thus the predicted accuracy for the out-of-sample error is 0.14%. This is a first-hand excellent result, also an expected outcome provided the type of the algorithm chosen, that did not invite to assess another algorithm. Moreover, the application of this model on the 20 test cases resulted in 100% prediction accuracy.

#### 6. System Information

```
##  sysname  release 
## "Darwin" "15.2.0"
```

```
## [1] "R version 3.1.2 (2014-10-31)"
```
