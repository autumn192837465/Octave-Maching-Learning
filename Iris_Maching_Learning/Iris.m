data = csvread('iris.csv');



## variables
fn = 4 + 1;   % feature number + ones
trainNum = 100;
testNum = 50;
dataNum = 150;
labelNum = 3; % 3 labels
lamda = 0.01;

r = randperm(150);
data = [ones(dataNum,1) data];


% 100 train data
dataTrain = data(r(1:trainNum),:);
dataTest = data(r(trainNum+1:end),:);

trainX = dataTrain(:,1:fn);
trainY = dataTrain(:,fn + 1);

testX = dataTest(:,1:fn);
testY = dataTest(:,fn + 1);

allTheta = OneVsAll(trainX,trainY,labelNum,lamda);
p = Predict(allTheta,testX);
fprintf('Accuracy: %f\n',mean(double(p == testY)) * 100);
