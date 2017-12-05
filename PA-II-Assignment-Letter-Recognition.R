library("ggplot2")
library("caret")
library("kernlab")
library("dplyr")
library("readr")
library("gridExtra")
library("caTools")

#Reading the datasets
digit_train<-read.csv('mnist_train.csv',stringsAsFactors = F,header = F)
digit_test<-read.csv('mnist_test.csv',stringsAsFactors = F,,header = F)

#Looking at the data
View(digit_train)
View(digit_test)

#Understanding the data
str(digit_train)
str(digit_test)

#Look at the dimension
dim(digit_train)
dim(digit_test)

## ------------------ Cleaning the datasets -----------------
# Looking for duplicate values
sum(duplicated(digit_train)) # No duplicates
sum(duplicated(digit_test))  # No duplicates

# Verify for Missing values
sum(is.na(digit_train)) # no NA values are present in training dataset
sum(is.na(digit_test))  # no NA values are present in testing  dataset

sum(sapply(digit_train, function(x) length(which(x == "")))) # no missing values
sum(sapply(digit_test, function(x) length(which(x == ""))))  # no missing values

#Exploring the data

summary(digit_train)
summary(digit_test) 

# Since the correct range of values are from 0 to 255 lets verify if any of the variable
# has out of range values
max(digit_train)#255 implies its within the range
max(digit_test)#255 implies its within the range
min(digit_train)#0 implies its within the range
min(digit_test)#0 implies its within the range

#Tried creating Linear and RBF models with 24,000, 15,000, 10,000 and 9,000 rows 
#selected out of 60,000 rows from digit_train dataset without removing any columns
#Got following issues:
#1. While doing 5 fold Cross Validation for Linear model got same value of c for all 
#the folds - 1 thru 5
#2. Also the execution time for hyperpameter tuning for RBF was exceptionally high. 

#Lets try doing removal of columns having same values across rows and PCA as the variable reduction methods in this case

set.seed(7)
index <- sample.split(digit_train$V1, SplitRatio =0.15)#15% of digit_train dataset
digit_train_final<-digit_train[index, ]

set.seed(7)
index1 <- sample.split(digit_test$V1, SplitRatio =0.20)#20% of digit_test dataset
digit_test_final<-digit_test[index1, ]


#Identify and remove the columns that have 0 values across all the rows in the train dataset
digit_train_final<-digit_train_final[,colSums(digit_train_final)>0] #Total variables=682

#Lets try doing PCA on the dataset digit_train_final
principal_component<-prcomp(digit_train_final[,-1],scale. = T)

standard_dev  <- principal_component$sdev
variance      <- standard_dev^2

feature_variance <- variance/sum(variance)

#Lets create a plot to have a look at the number variance
plot(feature_variance,xlab = "Principal Component",ylab = "Feature Variance Distribution",type = "b")

#Lets create training data post PCA, which will be used for model creation

digit_train_final_PCA <- data.frame(V1 = digit_train_final$V1, principal_component$x)

#Lets choose 50 principal components for this model
digit_train_final_PCA<-digit_train_final_PCA[,1:51]

#Lets perfrom PCA on the test data as well
digit_test_final_PCA <- predict(principal_component, newdata = as.data.frame(digit_test_final[,-1]))
digit_test_final_PCA <-as.data.frame(digit_test_final_PCA)#convert to a data frame
digit_test_final_PCA <-digit_test_final_PCA[,1:50]

#Lets convert output variable to factor type
digit_train_final_PCA$V1<-as.factor(digit_train_final_PCA$V1)
digit_test_final$V1<-as.factor(digit_test_final$V1)

#Lets apply Linear SVM
Model_Linear<-ksvm(V1~.,data=digit_train_final_PCA,scale = FALSE,kernel="vanilladot")

#Evaluating linear SVM
Eval_Linear<-predict(Model_Linear, digit_test_final_PCA)
confusionMatrix(Eval_Linear, digit_test_final$V1) #Got92% accuracy and below results
#############################################################################
#Accuracy : 0.92   
#Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
#Sensitivity            0.9745   0.9736   0.9272   0.9406   0.9490   0.8708   0.9479
#Specificity            0.9928   0.9944   0.9894   0.9850   0.9928   0.9874   0.9961
#Class: 7 Class: 8 Class: 9
#Sensitivity            0.9126   0.8359   0.8564
#Specificity            0.9900   0.9906   0.9928
#############################################################################

##################################Lets try RBF Kernel ###############################
#Using RBF Kernel
Model_RBF<-ksvm(V1~.,data=digit_train_final_PCA,scale = FALSE,kernel="rbfdot")

Eval_RBF<- predict(Model_RBF, digit_test_final_PCA)

confusionMatrix(Eval_RBF, digit_test_final$V1)#94.7% accuracy
###################################################################################
#Accuracy : 0.947
#Class: 0 Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
#Sensitivity            0.9796   0.9824   0.9515   0.9653   0.9592   0.9270   0.9583
#Specificity            0.9956   0.9955   0.9922   0.9917   0.9928   0.9934   0.9961
#Class: 7 Class: 8 Class: 9
#Sensitivity            0.9223   0.9077   0.9109
#Specificity            0.9950   0.9928   0.9961
###################################################################################

#Since we got better acuracy in RBF model, we will next do hyperparameters tuning
#using RBF kernel & 5 fold cross validation
trainControl <- trainControl(method="cv", number=5)

#Assign metric to accuracy
metric <- "Accuracy"

grid<-expand.grid(sigma=seq(0.01, 0.05, by=0.01), C=c(0.1,0.5,1,2))

#Performing 5-fold cross validation
fit.svm.RBF <- train(V1~.,data=digit_train_final_PCA, method="svmRadial", metric=metric, 
                     tuneGrid=grid, trControl=trainControl)

print(fit.svm.RBF)
# Best hyperpameters are : sigma = 0.03 & C=2

# Lets plot fit.svm.RBF
plot(fit.svm.RBF)

#Evaluating cross validation results against testing dataset
evaluate<- predict(fit.svm.RBF, digit_test_final_PCA)
confusionMatrix(evaluate, digit_test_final$V1)

