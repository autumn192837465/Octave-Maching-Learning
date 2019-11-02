function all_theta= OneVsAll (X, y, fn, lab_num,lamda)
  
  all_theta = zeros(fn,lab_num);  
  
  m = length(y);
    
  options = optimset('GradObj', 'on', 'MaxIter', 500000);    
  for i = 1:lab_num
    theta = rand(fn,1);
    all_theta(:,i) = fminunc(@(t)(CostFunction(t,X,(y == i),lamda)),theta,options);            
    
  end  
  #all_theta = all_theta(:,1);
endfunction



