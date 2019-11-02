function t = InitializeTheta (input_layer_num,hidden_layer_num,out_layer_num)
  e = 0.12;
  t_num = (input_layer_num + 1) * hidden_layer_num + (hidden_layer_num + 1) * out_layer_num;
  t = rand(t_num,1) * 2 * e - e;
endfunction
