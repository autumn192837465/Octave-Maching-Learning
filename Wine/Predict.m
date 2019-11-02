function p = Predict (X ,theta)
   #X*theta
   #p = Sigmoid(X*theta) > 0.5;
   #p = (Sigmoid(X*theta) > 0.5);
   [v,p] = max(X*theta,[],2);
endfunction
