library(readr) # reading csv
library(funModeling) # for EDA
library(DataExplorer)
library(mice)
library(dplyr)
library(forcats) 
library(MASS)# lda
library(earth)
library(caret)
library(Metrics)
install.packages("xgboost")
library(xgboost)
library(classifierplots)

?xgboost

#https://bradleyboehmke.github.io/HOML/logistic-regression.html#assessing-model-accuracy-1

hm7_Train <- read.csv("~/Desktop/SEM2/IDE/HW7/hm7-Train.csv")
hm7_Test <- read.csv("~/Desktop/SEM2/IDE/HW7/hm-7-Test.csv")

# taking traget variable aside and removing from train
Aoutput <- hm7_Train$readmitted

hm7_Train <- hm7_Train[-c(45)]
dim(hm7_Train)
dim(hm7_Test)

# comind data both test and train
total_data <- rbind(hm7_Train,hm7_Test)
dim(total_data)

funModeling::freq((total_data))


# EDA
funModeling::df_status(total_data)
summary(total_data)

# converting integer nomial values to factors, Becase they are not integer values they meant to be somthing else. So changeing to factors
total_data$admission_type <- as.factor(total_data$admission_type)
total_data$discharge_disposition <- as.factor(total_data$discharge_disposition)
total_data$admission_source <- as.factor(total_data$admission_source)


# checkng for missing ness, so that we can decide on imputation methods

funModeling::df_status((total_data))
funModeling::plot_num(total_data)


DataExplorer::plot_density(total_data)
# we can apply log for these cols as to reduce the skewness

total_data$num_lab_procedures <- log(total_data$num_lab_procedures)
total_data$num_medications <- log(total_data$num_medications)
total_data$num_procedures <- log(total_data$num_procedures+1)
total_data$number_diagnoses <-  log(total_data$number_diagnoses)
total_data$number_emergency <-  log(total_data$number_emergency+1)
total_data$number_inpatient <-  log(total_data$number_inpatient+1)
total_data$number_outpatient <-  log(total_data$number_outpatient+1)
total_data$time_in_hospital <-  log(total_data$time_in_hospital)

DataExplorer::plot_density(total_data)

colSums(is.na(total_data))


#as.character( Mode(total_data$race) )

total_data$race <- fct_explicit_na(total_data$race, na_level = '(missing)' )

total_data$payer_code <- fct_explicit_na(total_data$payer_code, na_level = '(missing)')

total_data$medical_specialty <- fct_explicit_na(total_data$medical_specialty, na_level = '(missing)')

total_data$diagnosis <- fct_explicit_na(total_data$diagnosis, na_level = '(missing)')


colSums(is.na(total_data))
summary(total_data)

# removing '-' in medical speciality for some reason it is throwing errors

total_data$medical_specialty <- as.factor(as.numeric(total_data$medical_specialty))

total_data$age <- as.factor(as.numeric(total_data$age))
total_data$max_glu_serum <- as.factor(as.numeric(total_data$max_glu_serum))
total_data$A1Cresult <- as.factor(as.numeric(total_data$A1Cresult))

total_data$visits = total_data$number_outpatient + total_data$number_emergency + total_data$number_inpatient

# we can remove 36,37  which has only one value and 41 col has only one value 'no' in train data but has 'steady' value in test data
# id also removing is has all unique values only
# payer_code - which has more missing data - feels irrelevent to analysis (9)

total_data <- total_data[ -c(1,36,37) ]

# -------------------------------------------------- features

## Detecting columns with low to zero variance

#low_var_features <- nearZeroVar(total_data, names = T, freqCut = 19, uniqueCut = 10)
#low_var_features

#total_data <- total_data[, -which(names(total_data) %in% low_var_features)]

# -------------------------------------------------------------------  setting train and test data
dim(total_data)
traindata <- total_data[1:57855,]
dim(traindata)
traindata$Readmitted <- as.factor(Aoutput)
dim(traindata)
testdata <- total_data[57856:96423,]
dim(testdata)


# -------------------------------------------------------------------  Modeling

# ------------------------------------------------------------------- logistic regression

set.seed(123)
logistic <- train(
  Readmitted ~ ., 
  data = traindata, 
  method = "glm",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 3)
)
?glm
logistic$results

m1_prob <- predict(logistic, testdata, type = "prob")$`1`

logLoss(as.numeric(traindata$Readmitted), predict(logistic, traindata, type = "prob")$`1`)
# 0.689 

logisticres<-tibble(patientID=hm7_Test$patientID, predReadmit= m1_prob)
head(logisticres)
write.csv(logisticres, file = paste("~/Desktop/SEM2/IDE/HW7/submitlogisticres",Sys.time(),".csv"), row.names=FALSE)
# got 0.66 score after kaggle submission

# ----------------------------  LASSO
set.seed(123) 
library(glmnet)
lasso <- train(
  Readmitted ~ .,   # model to fit
  data = traindata, 
  method = "glmnet",
  family = 'binomial',
  trControl = trainControl("cv", number = 3),
  tuneGrid = expand.grid(alpha = 1, lambda = seq(0.01,0.1 ,length = 100))
)



logLoss(as.numeric(traindata$Readmitted), predict(lasso, traindata, type = "prob")$`1`)


m1.1_prob <- predict(lasso, testdata, type = "prob")$`1`






# ------------------------------------------------------------------- MARS regression

# for reproducibiity
set.seed(123)
hyper_grid <- expand.grid(
  degree = 1:3, 
  nprune = seq(2, 100, length.out = 10) %>% floor()
)
# cross validated model
tuned_mars <- caret::train(
  Readmitted ~ .,
  data = traindata,
  method = "earth",
  trControl = caret::trainControl(method = "cv", number = 3),
  tuneGrid = hyper_grid,
  glm=list(family=binomial)
)
?earth
# best model
tuned_mars$bestTune
tuned_mars$modelType
tuned_mars$resample
tuned_mars$finalModel$coefficients
tuned_mars$
  
  
marslogloss <- logLoss(ifelse(traindata$Readmitted=="1",1,0), predict(tuned_mars, traindata, type = "prob")$`1`)
# loss - 0.640
m2_prob <- predict(tuned_mars, testdata, type = "prob")$`1`
ptuned_marsres<-tibble(patientID=hm7_Test$patientID, predReadmit= m2_prob)
head(ptuned_marsres)
write.csv(ptuned_marsres, file = paste("~/Desktop/SEM2/IDE/HW7/submitMars",Sys.time(),".csv"), row.names=FALSE)

tuned_mars$results

# ------------------------------------------------------------------- decision trees
library(rpart)
dtraindata = traindata
dtraindata$Readmitted <- as.factor(ifelse(traindata$Readmitted=="1",'yes','no'))

cpGrid <- expand.grid(.cp=seq(2,43,2))
# caret cross validation results
system.time(decisiontrees <- train(
  Readmitted ~ .,
  data = dtraindata,
  method = "rpart",
  trControl = trainControl(method = "cv", number = 3,classProbs = TRUE),
  tuneGrid = cpGrid
))

?rpart
plot(decisiontrees)
decisiontrees$bestTune


decisiontreeslogloss <- logLoss(ifelse(dtraindata$Readmitted=="yes",1,0), predict(decisiontrees, dtraindata, type = "prob")$yes)
# 0.6582 loss
m3_prob <- predict(decisiontrees, testdata, type = "prob")$yes
ptuned_decision<-tibble(patientID=hm7_Test$patientID, predReadmit= m3_prob)
head(ptuned_decision)
write.csv(ptuned_decision, file = paste("~/Desktop/SEM2/IDE/HW7/submit_ptuned_decision",Sys.time(),".csv"), row.names=FALSE)
decisiontrees$results


decisiontrees

#------------- below one works best

set.seed(123)
model <- train(
  Readmitted ~., 
  data = dtraindata, 
  method = "rpart",
  metric = "logLoss",
  trControl = trainControl("cv", number = 5),
  tuneLength = 10
)

logLoss(ifelse(dtraindata$Readmitted=="yes",1,0), as.data.frame(predict(model, dtraindata, type = "prob"))$yes)



# ------------------------------------------------------------------- random forests

set.seed(123)

rftotaldata = total_data
rftotaldata$diagnosis <- fct_lump(total_data$diagnosis , n = 50)
rftotaldata$medical_specialty <- fct_lump(total_data$medical_specialty , n = 50)
rftraindata <- rftotaldata[1:57855,]
rftraindata$Readmitted <- as.factor(Aoutput)
rftestdata <- rftotaldata[57856:96423,]

result <- randomForest::rfcv(rftraindata[-c(43)], rftraindata$Readmitted, cv.fold=3)



control <- trainControl(method="cv", number=3)
tunegrid <- expand.grid(.mtry=sqrt(42))
system.time(rf_random <- train(Readmitted~., 
                               data=rftraindata, 
                               method="rf",
                               tuneGrid=tunegrid, 
                               trControl=control))

randomloss <- logLoss(ifelse(dtraindata$Readmitted=="yes",1,0), predict(rf_random, dtraindata, type = "prob")$yes)
# loss 
?randomForest::rfcv
m4_prob <- predict(rf_random, testdata, type = "prob")$yes
randomforestpre<-tibble(patientID=hm7_Test$patientID, predReadmit= m4_prob)
head(randomforestpre)
write.csv(randomforestpre, file = paste("~/Desktop/SEM2/IDE/HW7/submit_randomforest",Sys.time(),".csv"), row.names=FALSE)


# ------------------------------------------------------------------- xgboost
#https://analyticsdataexploration.com/xgboost-model-tuning-in-crossvalidation-using-caret-in-r/

parametersGrid <-  expand.grid(eta = 0.05, 
                               colsample_bytree=c(1,0.75),
                               max_depth=c(1:10),
                               nrounds=500,
                               gamma=0,
                               min_child_weight=0,
                               subsample = c(0.5,0.75,1)
)

ControlParamteres <- trainControl(method = "cv",
                                  number = 3,
                                  savePredictions = TRUE,
                                  classProbs = TRUE
)

system.time(modelxgboost <- train(Readmitted~., 
                      data = dtraindata,
                      method = "xgbTree",
                      trControl = ControlParamteres,
                      tuneGrid=parametersGrid))
# working great
xgboostlogloss <- logLoss(ifelse(dtraindata$Readmitted=="yes",1,0), predict(modelxgboost, dtraindata, type = "prob")$yes)
# 0.63 loss


m5_prob <- predict(modelxgboost, testdata, type = "prob")$yes
xgboostpre<-tibble(patientID=hm7_Test$patientID, predReadmit= m5_prob)
head(xgboostpre)
write.csv(xgboostpre, file = paste("~/Desktop/SEM2/IDE/HW7/submit_xgboost",Sys.time(),".csv"), row.names=FALSE)


modelxgboost$results
modelxgboost$bestTune 
modelxgboost$modelInfo$varImp
plot(modelxgboost)
gbmImp <- varImp(modelxgboost, scale = TRUE)


maxpred <- as.data.frame(predict(modelxgboost, dtraindata, type = "prob"))$yes

class_prediction <-
  as.factor(ifelse(maxpred > 0.5,
                   'yes',
                   'no'
  ))
cm <- confusionMatrix(class_prediction,dtraindata$Readmitted)

cm$byClass

library('pROC')
rocdata <- roc(ifelse(dtraindata$Readmitted=="yes",1,0),maxpred )
title('ROC Curve')
auc(rocdata)



calibration_plot
classifierplots::calibration_plot(ifelse(dtraindata$Readmitted=="yes",1,0),maxpred)

density_plot
classifierplots::density_plot(ifelse(dtraindata$Readmitted=="yes",1,0),maxpred)

lift_plot
classifierplots::roc_plot(ifelse(dtraindata$Readmitted=="yes",1,0),maxpred)


# ------------------------------------------------------------------- svm
#http://dataaspirant.com/2017/01/19/support-vector-machine-classifier-implementation-r-caret-package/
system.time(svm_Radial <- train(Readmitted ~., data = training, method = "svmRadial",
                    trControl= trainControl(method="cv", number=3, classProb=T),
                    preProcess = c("center", "scale"),
                    tuneLength = 10))

grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04,
                                       0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
                             C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                                   1, 1.5, 2,5))
set.seed(3233)
system.time(svm_Radial_Grid <- train(V14 ~., data = training, method = "svmRadial",
                           trControl=trainControl(method="cv", number=3, classProb=T),
                           preProcess = c("center", "scale"),
                           tuneGrid = grid_radial,
                           tuneLength = 10))
#loss



# -------------------------------------------------------------------
# cleaning
#rm(list = ls(all.names = TRUE))
#gc()




