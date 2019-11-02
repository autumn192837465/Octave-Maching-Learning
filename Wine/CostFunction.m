function [J, grad] = CostFunction (theta, X, y, lamda)
  m = size(X,1);
  grad = zeros(size(theta));
  
  J = (-y' * log(Sigmoid(X * theta)) - (1 - y)' * log(1 - Sigmoid(X * theta)))/ m + lamda * (theta(2:end)' * theta(2:end)) / 2 / m;  
  grad = (X' * (Sigmoid(X * theta) - y))/m + lamda * [0; theta(2:end)] / m;
endfunction
