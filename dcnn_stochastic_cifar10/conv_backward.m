function [DLDx,DLDy,DLDz]=conv_backward(DLDp,p_idx,DzDy,conv)
% backward pass for convolution layer

DLDz=DzDy;
DLDz(:)=0;
DLDz(p_idx)=DLDp;

DLDy=DLDz.*DzDy;

kernels_back=permute(rot90(conv.weight,2),[1,2,4,3]);

switch conv.type
    case 'valid'
        DLDx=dlconv(DLDy,kernels_back,0,'Padding',(conv.n_kernel-1));
    case 'same'
        DLDx=dlconv(DLDy,kernels_back,0,'Padding','same');
end