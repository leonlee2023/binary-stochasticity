function [y,z,DzDy,p,p_idx]=conv_forward(x,conv)
% forward pass for convolution and pooling layer

switch conv.type
    case 'valid'
        y=dlconv(x,conv.weight,conv.biases);
    case 'same'
        y=dlconv(x,conv.weight,conv.biases,'Padding','same');
end

[z,DzDy]= activation(y,conv.act_a); 

[p,p_idx]=maxpool(z,conv.n_pool,'Stride',conv.n_pool);


