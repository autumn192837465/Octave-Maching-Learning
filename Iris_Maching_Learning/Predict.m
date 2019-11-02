function p = onePredictAll (theta, X)
  m = size(X,1);
  [A,p] = max(Sigmoid(X * theta),[],2);
  p -= 1;
endfunction
