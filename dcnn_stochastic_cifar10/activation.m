function [z,DzDy]= activation(y,a)
% activation function, and derivative of activation function to its input

    z = 1./(1+exp(-a*y));
    DzDy = a*z.*(1-z);
end