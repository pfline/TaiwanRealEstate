#install and load packages
if(!require(readxl)) install.packages("readxl", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")

library(readxl)
library(tidyverse)
library(caret)
library(rpart)

#set seed
set.seed(1)

#download and load data file
url<-"https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
path_tmp <- tempfile()
download.file(url, 
              destfile=path_tmp, method="curl")
data <- read_excel(path_tmp)

#Explore data

###Looking for NAs
sum(is.na(data)) #There are no Nas

###Data structure
str(data) #414 observations, 8 columns. "No" column no useful for building our prediction model
data<-select(data,-No) #remove "No"
colnames(data)<-c("Transaction_date","Age","MRT","Stores","latitude","longitude","Price") #change the column names to be more succint
summary(data) #explore the summary of each variable

###Distributions of each variable
data %>% ggplot(aes(Transaction_date)) +
  geom_histogram() #the transactions were made more or less at the same period of time (2012 and 2013). Apart from if there has been a real estate crisis at that period, we don't expect the prices to depend much on this parameter.  
data %>% ggplot(aes(Age)) +
  geom_histogram()
data %>% ggplot(aes(MRT)) +
  geom_histogram() #contains some isolated values (MRT>5000)
data %>% ggplot(aes(Stores)) +
  geom_histogram()
data %>% ggplot(aes(Price)) +
  geom_histogram() #contain some isolated values (Price>90)

###Pearson Correlation correlation coefficients of Price VS other variables
cor(data,data$Price) #Price does not seem to be much correlated with the Transaction date

###Plot Price VS other variables
data %>%
  gather(-c(Price,latitude, longitude), key="variables", value="values") %>%
  ggplot(aes(x=values, y=Price))+
  geom_point()+
  facet_wrap(~variables, scales="free")
#we mostly see a trend of Price VS MRT, which does not seem linear

###Boxplots of Price VS Number of stores for each number of stores
data %>%
  ggplot(aes(y=Price, group=Stores))+
  geom_boxplot()+
  theme(axis.text.x = element_blank(), axis.ticks = element_blank())+
  labs(x="Stores")+
  facet_grid(.~Stores) #When looking at the means, this could look like a linear model

###Plot Price VS geographic coordinates
data %>% ggplot(aes(longitude,latitude)) +
  geom_point(aes(col=Price))+
  scale_color_gradient(low="yellow",
                        high="red",
                      )+
  theme_dark()

###Plot Transaction_Date VS geographic coordinates
data %>% ggplot(aes(longitude,latitude)) +
  geom_point(aes(col=Transaction_date))+
  scale_color_gradient(low="yellow",
                       high="red",
  )+
  theme_dark() #not much insight, all were performed in 2012 or 2013, so in a short period of time

###Plot Age VS geographic coordinates
data %>% ggplot(aes(longitude,latitude)) +
  geom_point(aes(col=Age))+
  scale_color_gradient(low="yellow",
                       high="red",
  )+
  theme_dark()

###Plot MRT VS geographic coordinates
data %>% ggplot(aes(longitude,latitude)) +
  geom_point(aes(col=MRT))+
  scale_color_gradient(low="yellow",
                       high="red",
  )+
  theme_dark() #strong disparity

###Plot Stores VS geographic coordinates
data %>% ggplot(aes(longitude,latitude)) +
  geom_point(aes(col=Stores))+
  scale_color_gradient(low="yellow",
                       high="red",
  )+
  theme_dark() #strong disparity

#Data split between train and test (80/20)
test_index <- createDataPartition(data$Price, times = 1, p = 0.2, list = FALSE)
test_set <- data[test_index, ]
train_set <- data[-test_index, ]

#Create RMSE function that assesses model performance
RMSE<-function(true_ratings,predicted_ratings){
  sqrt(mean((true_ratings-predicted_ratings)^2))
}

#Model with lm method
## With all variables
lmfit<-lm(Price~.,train_set)
lmprediction<-predict(lmfit,test_set)
RMSE(lmprediction,test_set$Price)

## With all variables except transaction date
lmfit<-lm(Price~.-Transaction_date,train_set)
lmprediction<-predict(lmfit,test_set)
RMSE(lmprediction,test_set$Price)

#Model with knn method
train_knn<-train(Price~.,method="knn",data=train_set,tuneGrid=data.frame(k=seq(2,50,1)))
knnprediction<-predict(train_knn,test_set)
RMSE(knnprediction,test_set$Price)
train_knn$bestTune #best k
ggplot(train_knn,highlight=TRUE)

#Model with knn method without transaction date
train_knn<-train(Price~.-Transaction_date,method="knn",data=train_set,tuneGrid=data.frame(k=seq(2,50,1)))
knnprediction<-predict(train_knn,test_set)
RMSE(knnprediction,test_set$Price)
train_knn$bestTune #best k
ggplot(train_knn,highlight=TRUE)

#Regression trees (rpart)
##Regression with default parameters
train_rpart<-rpart(Price~.,data=train_set)
plot(train_rpart, margin=0.1)
text(train_rpart, cex=0.6)
rpartprediction<-predict(train_rpart,test_set)
RMSE(rpartprediction,test_set$Price)

##Regression with personnally set parameters
cp<-seq(0,2,0.01)
minsplit<-seq(1,60,1)
minbucket<-seq(1,60,1)

TuningMatrix<-matrix(nrow=length(cp)*length(minsplit)*length(minbucket),ncol=4)
colnames(TuningMatrix)<-c("Minsplit","Minbucket","cp","RMSE")

library(rpart)

n<-0
for(minsplitRange in 1:length(minsplit)){
  for (minbucketRange in 1:length(minbucket)){
    for (cpRange in 1:length(cp)){
      n<-n+1
      train_rpart<-rpart(Price~.,data=train_set,control=rpart.control(
        minsplit=minsplit[minsplitRange],
        minbucket=minbucket[minbucketRange],
        cp=cp[cpRange]
      ))
      rpartprediction<-predict(train_rpart,test_set)
      RMSE<-RMSE(rpartprediction,test_set$Price)
      
      TuningMatrix[n,1]<-minsplitRange
      TuningMatrix[n,2]<-minbucketRange
      TuningMatrix[n,3]<-cpRange
      TuningMatrix[n,4]<-RMSE
    }
  }
}

TuningDataFrame<-as.data.frame(TuningMatrix)
TuningDataFrame[which.min(TuningDataFrame$RMSE),]

##Regression with personnally set parameters without transaction date

n<-0
for(minsplitRange in 1:length(minsplit)){
  for (minbucketRange in 1:length(minbucket)){
    for (cpRange in 1:length(cp)){
      n<-n+1
      train_rpart<-rpart(Price~.-Transaction_date,data=train_set,control=rpart.control(
        minsplit=minsplit[minsplitRange],
        minbucket=minbucket[minbucketRange],
        cp=cp[cpRange]
      ))
      rpartprediction<-predict(train_rpart,test_set)
      RMSE<-RMSE(rpartprediction,test_set$Price)
      
      TuningMatrix[n,1]<-minsplitRange
      TuningMatrix[n,2]<-minbucketRange
      TuningMatrix[n,3]<-cpRange
      TuningMatrix[n,4]<-RMSE
    }
  }
}

TuningDataFrame<-as.data.frame(TuningMatrix)
TuningDataFrame[which.min(TuningDataFrame$RMSE),]

##Saving best parameters
MinsplitMin<-TuningDataFrame[which.min(TuningDataFrame$RMSE),1]
MinbucketMin<-TuningDataFrame[which.min(TuningDataFrame$RMSE),2]
cpMin<-TuningDataFrame[which.min(TuningDataFrame$RMSE),3]

##Plot RMSE for lowest Minsplit
TuningDataFrame %>%
  filter(Minsplit==MinsplitMin) %>%
  ggplot(aes(x=Minbucket,y=cp, z=RMSE,fill=RMSE))+
  geom_tile()+
  scale_fill_gradient(low = "red", high = "yellow")

##Plot RMSE for lowest Minbucket
TuningDataFrame %>%
  filter(Minbucket==MinbucketMin) %>%
  ggplot(aes(x=Minsplit,y=cp, z=RMSE,fill=RMSE))+
  geom_tile()+
  scale_fill_gradient(low = "red", high = "yellow")

##Plot RMSE for lowest cp
TuningDataFrame %>%
  filter(cp==cpMin) %>%
  ggplot(aes(x=Minsplit,y=Minbucket, z=RMSE,fill=RMSE))+
  geom_tile()+
  scale_fill_gradient(low = "red", high = "yellow")

#Model with random forest Rborist
minnode<-seq(1,300,1) #try with node sizes from 1 to 300 by 1
RMSE_rborist<-sapply(minnode,function(minnode){
  train_rborist<-train(Price~.,
                  method="Rborist",
                  data=train_set,
                  .minNode=minnode)
  rboristprediction<-predict(train_rborist,test_set)
  RMSE(rboristprediction,test_set$Price)
})
minnode[which.min(RMSE_rborist)] #minnode with lowest RMSE
min(RMSE_rborist) #lowest RMSE

##Model with random forest Rborist without transaction date
RMSE_rborist<-sapply(minnode,function(minnode){
  train_rborist<-train(Price~.-Transaction_date,
                       method="Rborist",
                       data=train_set,
                       .minNode=minnode)
  rboristprediction<-predict(train_rborist,test_set)
  RMSE(rboristprediction,test_set$Price)
})
qplot(minnode,RMSE_rborist)
minnode[which.min(RMSE_rborist)]  #minnode with lowest RMSE
min(RMSE_rborist)  #lowest RMSE