%%%%% Load data
data = load('wine.csv');


%%%% variables
m = size(data,1);
fn = size(data,2);
lamda = 0.1;


%%%% seperate data
r = randperm(m);
data = data(r(:),:);

X = [ones(m,1) data(:,2:end)];
y = data(:,1);

idx = round(m * 0.7);
trainX = X(1:idx,:);
trainY = y(1:idx);
testX = X(idx+1:end,:);
testY = y(idx+1:end);

%%% how many classes
class = length(unique(y));

%theta = 


theta = OneVsAll(trainX,trainY,fn,class,lamda);
#theta = rand(fn,1);
#theta = fminunc(@(t)(CostFunction(t,trainX, trainY == 1,0)),theta,options);    
#testX * theta
p = Predict(testX,theta);
#a = [p testY]

accuracy = mean(double(p == testY)) * 100;
fprintf('\nAccuracy: %f\n', accuracy);

# Plot

figure(1);
bar(p,'LineStyle','none',"FaceColor",[0 0.75 0.75]);
hold on
plot(testY,"bo");
legend('Actual','Predict');
xlabel('Data number');
ylabel('Class');

annotation('textbox',[.91 .5 .1 .2],'String', strcat('Accuracy : ',num2str(accuracy)) ,'EdgeColor','none')
