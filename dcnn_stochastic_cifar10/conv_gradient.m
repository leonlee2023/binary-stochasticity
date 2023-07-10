function DLDw=conv_gradient(x,DLDy,conv)
% computing the gradient of the convolutional layer

x=dlarray(x,'SSUC');
DLDy=dlarray(DLDy,'SSUC');
switch conv.type
    case 'valid'
        DLDw=dlconv(x,DLDy,0);
    case 'same'
        DLDw=dlconv(x,DLDy,0,'Padding',(conv.n_kernel-1)/2);
end
DLDw=permute(stripdims(DLDw),[1,2,4,3])/(conv.input_size^2);

