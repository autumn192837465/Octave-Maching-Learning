%%%% Load data %%%%
data = load('Iris.csv');

%% variable
fn = size(data,2);
m = size(data,1);
input_layer_num = fn-1;
hidden_layer_num = 15;
output_layer_num = 3;
lambda = 1;

%% seperate X y
X = [ones(m,1) data(:,1:fn-1)];
y = data(:,fn);




%% train data and test data
r = randperm(m);
trainX = X(r(1:round(m * 0.7)),:);
trainY = y(r(1:round(m * 0.7)),1);
testX = X(r(round(m*0.7+1) : end),:);
testY = y(r(round(m*0.7+1) : end),1);





%% Initialize theta
initialTheta = InitializeTheta(input_layer_num,hidden_layer_num,output_layer_num);

options = optimset('GradObj','on','MaxIter',100000);


nnCostFunction = @(t)CostFunction(t,trainX,trainY,output_layer_num,input_layer_num,hidden_layer_num,lambda);
[theta,j] = fmincg(nnCostFunction,initialTheta,options);

theta1 = reshape(theta(1:(input_layer_num + 1) * hidden_layer_num),hidden_layer_num,input_layer_num+1);
theta2 = reshape(theta((input_layer_num + 1) * hidden_layer_num + 1 : end), output_layer_num,hidden_layer_num+1);

p = Predict(theta1,theta2,testX,testY);

fprintf("Accuracy %.2f ",mean(double(p == testY))* 100);


