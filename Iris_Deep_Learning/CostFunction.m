function [J grad]= CostFunction (theta, X, y, label_num, input_layer_num,hidden_layer_num, lambda)
  m = length(y);
  
  
  theta1 = reshape(theta(1:(input_layer_num + 1) * hidden_layer_num),hidden_layer_num, input_layer_num+1);
  theta2 = reshape(theta((input_layer_num + 1) * hidden_layer_num + 1: end), label_num, hidden_layer_num +1);
  
  
  
  a1 = X;
  z2 = a1 * theta1';
  a2 = [ones(m,1) Sigmoid(z2)];
  z3 = a2 * theta2';
  a3 = Sigmoid(z3);   
     
  yk = zeros(m,label_num);
  for i = 1:m
    yk(i,y(i)+1) = 1;
  end  


  
  %% Cost
  J = sum(sum((-yk) .* log(a3) - (1 - yk) .* log(1 - a3)))/m + (sum(sum(theta2(:,2:end).^2)) + sum(sum(theta1(:,2:end).^2))) * lambda / (2 * m);
  
  delta3 = a3 - yk;
  delta2 = delta3 * theta2 .* [ones(m,1) GradSigmoid(z2)];
  delta2 = delta2(:,2:end);
  
  theta1Grad = delta2' * a1 / m + lambda * [zeros(hidden_layer_num,1) theta1(:,2:end)] / m;
  theta2Grad = delta3' * a2 / m + lambda * [zeros(label_num,1) theta2(:,2:end)] / m;
  
  grad = [theta1Grad(:) ; theta2Grad(:)];     
  
endfunction
