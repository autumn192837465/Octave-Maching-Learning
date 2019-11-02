function p = Predict (t1,t2 , X, y)
  
  m = length(y);
  
  h1 = [ones(m,1) Sigmoid(X * t1')];
  h2 = Sigmoid(h1 * t2');
  [dummy p] = max(h2,[],2);
  p -= 1;
endfunction
