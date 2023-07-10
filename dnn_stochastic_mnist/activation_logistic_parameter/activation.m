function [z,dzdy]= activation(y,a,d)
% activation function, and derivative of activation function to its input

    z = 1./(1+exp(-a*y));
    dzdy = d*z.*(1-z);

end