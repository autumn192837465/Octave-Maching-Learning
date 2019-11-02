function g = GradSigmoid (z)
  g = (1 - Sigmoid(z)) .* Sigmoid(z);
endfunction
