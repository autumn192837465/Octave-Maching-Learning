function allTheta = OneVsAll (X, y, labelNum, lamda)
  m = size(X,1);
  n = size(X,2);
  
  allTheta = zeros(n,labelNum);
  
  %  Set options for fminunc
  options = optimset('GradObj', 'on', 'MaxIter', 1000);  
  
  initial_theta = zeros(n,1);  
  for i = 1:labelNum   
    allTheta(:,i) = fminunc(@(t)(CostFunction(t, X, (y == (i-1)),lamda)), initial_theta, options);
  end
endfunction
