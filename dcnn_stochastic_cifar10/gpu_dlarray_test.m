clc
clear
close all

height=28;  
width=28;
batchsize=100;
channel_x=32;
channel_y=64;
k=5;  % should be an odd number 

% %%%%%%%%%% running on GPU %%%%%%%%%%%%%%%
% gpurng('default')
% fprintf('############ runing on GPU ############')
% x=randn(height,width,channel_x,batchsize,'single','gpuArray');
% dx=zeros(height,width,channel_x,batchsize,'single','gpuArray');
% x_extended=zeros(height+(k-1),width+(k-1),channel_x,batchsize,'single','gpuArray');
% x_extended((k-1)/2+(1:height),(k-1)/2+(1:width),:,:)=x;
% kernels=randn(k,k,channel_x,channel_y,'single','gpuArray');
% y=zeros(height,width,channel_y,batchsize,'single','gpuArray');
% dy=randn(height,width,channel_y,batchsize,'single','gpuArray');
% DLDw=zeros(size(kernels),'single','gpuArray');

% % convolutional layer forwarding
% tic
% fprintf("\ngpuArray: forwarding through convolutional layer... \n")
% for out = 1:channel_y
%     for in = 1:channel_x
%         kernel=kernels(:,:,in,out);
%         y(:,:,out,:)=y(:,:,out,:)+convn(x(:,:,in,:),rot90(kernel,2),'same');
%     end
% end
% fprintf(' y=x*w:\n')
% fprintf(' x(SSCB): (%d, %d, %d, %d)\n',size(x))
% fprintf(' w(SSCxCy): (%d, %d, %d, %d)\n',size(kernels))
% fprintf(' y(SSCB): (%d, %d, %d, %d)\n',size(y))
% toc

% % convolutional layer error backpropagatiing
% fprintf("\ngpuArray: backpropagating through convolutional layer... \n")
% tic
% for out = 1:channel_y
%     for in = 1:channel_x
%         kernel=kernels(:,:,in,out);
%         dx(:,:,in,:)=dx(:,:,in,:)+convn(dy(:,:,out,:),kernel,'same');
%     end
% end
% fprintf(' dx=dy*wT:\n')
% fprintf(' dy(SSCB): (%d, %d, %d, %d)\n',size(dy))
% fprintf(' w(SSCxCy): (%d, %d, %d, %d)\n',size(kernels))
% fprintf(' dx(SSCB): (%d, %d, %d, %d)\n',size(dx))
% toc

% % griadent of the loss to the weights
% fprintf("\ngpuArray: computing griadent in the convolutional layer... \n")
% tic
% for out = 1:channel_y
%     for in = 1:channel_x
%         for bb=1:batchsize
%             DLDw(:,:,in,out)=DLDw(:,:,in,out)+convn(x_extended(:,:,in,bb),rot90(dy(:,:,out,bb),2),'valid');
%         end
%     end
% end
% fprintf(' DLDw=x*dy:\n')
% fprintf(' x_extended(SSCB): (%d, %d, %d, %d)\n',size(x_extended))
% fprintf(' dy(SSCB): (%d, %d, %d, %d)\n',size(dy))
% fprintf(' DLDw(SSCxCy): (%d, %d, %d, %d)\n',size(DLDw))
% toc


% % convolutional layer forwarding
% tic
% fprintf("\n\ngpu+dlarray: forwarding through convolutional layer... \n")
% X=dlarray(x,'SSCB');
% Y=dlconv(X,kernels,0,'Padding','same');
% 
% fprintf(' Y=X*w:\n')
% fprintf(' X(SSCB): (%d, %d, %d, %d)\n',size(X))
% fprintf(' w(SSCxCy): (%d, %d, %d, %d)\n',size(kernels))
% fprintf(' Y(SSCB): (%d, %d, %d, %d)\n',size(Y))
% toc
% 
% Z=sigmoid(Y);
% [P,indx]=maxpool(Z,2,'Stride',2);
% 
% % convolutional layer error backpropagatiing
% tic
% fprintf("\ngpu+dlarray: backpropagating through convolutional layer... \n")
% dY=dlarray(dy,'SSCB');
% kernels_back=permute(rot90(kernels,2),[1,2,4,3]);
% dX=dlconv(dY,kernels_back,0,'Padding','same');
% 
% fprintf(' dX=dY*wT:\n')
% fprintf(' dY(SSCB): (%d, %d, %d, %d)\n',size(dY))
% fprintf(' w(SSCyCx): (%d, %d, %d, %d)\n',size(kernels_back))
% fprintf(' dX(SSCB): (%d, %d, %d, %d)\n',size(dX))
% toc
% 
% % graident of the loss to the weights
% fprintf("\ngpu+dlarray: computing griadent in the convolutional layer... \n")
% X_extended=dlarray(x_extended,'SSUC');
% dY=dlarray(dy,'SSUC');
% DLDW=dlconv(X_extended,dY,0);
% DLDW=permute(stripdims(DLDW),[1,2,4,3]);
% fprintf(' DLDW=X*dY:\n')
% fprintf(' X_extended(SSCB): (%d, %d, %d, %d)\n',size(X_extended))
% fprintf(' dY(SSCB): (%d, %d, %d, %d)\n',size(dY))
% fprintf(' DLDW(SSCxCy): (%d, %d, %d, %d)\n',size(DLDW))
% toc
% 
% fprintf('\n\n')
%%%%%%%%%% running on CPU %%%%%%%%%%%%%%%
rng('default')
fprintf('############ runing on CPU ############')
x=randn(height,width,channel_x,batchsize,'single');
dx=zeros(height,width,channel_x,batchsize,'single');
x_extended=zeros(height+(k-1),width+(k-1),channel_x,batchsize,'single');
x_extended((k-1)/2+(1:height),(k-1)/2+(1:width),:,:)=x;
kernels=randn(k,k,channel_x,channel_y,'single');
y=zeros(height,width,channel_y,batchsize,'single');
dy=randn(height,width,channel_y,batchsize,'single');
DLDw=zeros(size(kernels),'single');

% convolutional layer forwarding
tic
fprintf("\ncpu: forwarding through convolutional layer... \n")
for out = 1:channel_y
    for in = 1:channel_x
        kernel=kernels(:,:,in,out);
        y(:,:,out,:)=y(:,:,out,:)+convn(x(:,:,in,:),rot90(kernel,2),'same');
    end
end
fprintf(' y=x*w:\n')
fprintf(' x(SSCB): (%d, %d, %d, %d)\n',size(x))
fprintf(' w(SSCxCy): (%d, %d, %d, %d)\n',size(kernels))
fprintf(' y(SSCB): (%d, %d, %d, %d)\n',size(y))
toc

% convolutional layer error backpropagatiing
fprintf("\ncpu: backpropagating through convolutional layer... \n")
tic
for out = 1:channel_y
    for in = 1:channel_x
        kernel=kernels(:,:,in,out);
        dx(:,:,in,:)=dx(:,:,in,:)+convn(dy(:,:,out,:),kernel,'same');
    end
end
fprintf(' dx=dy*wT:\n')
fprintf(' dy(SSCB): (%d, %d, %d, %d)\n',size(dy))
fprintf(' w(SSCxCy): (%d, %d, %d, %d)\n',size(kernels))
fprintf(' dx(SSCB): (%d, %d, %d, %d)\n',size(dx))
toc

% griadent of the loss to the weights
fprintf("\ncpu: computing griadent in the convolutional layer... \n")
tic
for out = 1:channel_y
    for in = 1:channel_x
        for bb=1:batchsize
            DLDw(:,:,in,out)=DLDw(:,:,in,out)+convn(x_extended(:,:,in,bb),rot90(dy(:,:,out,bb),2),'valid');
        end
    end
end
fprintf(' DLDw=x*dy:\n')
fprintf(' x_extended(SSCB): (%d, %d, %d, %d)\n',size(x_extended))
fprintf(' dy(SSCB): (%d, %d, %d, %d)\n',size(dy))
fprintf(' DLDw(SSCxCy): (%d, %d, %d, %d)\n',size(DLDw))
toc



% convolutional layer forwarding
tic
fprintf("\n\ncpu+dlarray: forwarding through convolutional layer... \n")
X=dlarray(x,'SSCB');
Y=dlconv(X,kernels,0,'Padding','same');

fprintf(' Y=X*w:\n')
fprintf(' X(SSCB): (%d, %d, %d, %d)\n',size(X))
fprintf(' w(SSCxCy): (%d, %d, %d, %d)\n',size(kernels))
fprintf(' Y(SSCB): (%d, %d, %d, %d)\n',size(Y))
toc

Z=sigmoid(Y);
[P,indx]=maxpool(Z,2,'Stride',2);

% convolutional layer error backpropagatiing
tic
fprintf("\ncpu+dlarray: backpropagating through convolutional layer... \n")
dY=dlarray(dy,'SSCB');
kernels_back=permute(rot90(kernels,2),[1,2,4,3]);
dX=dlconv(dY,kernels_back,0,'Padding','same');

fprintf(' dX=dY*wT:\n')
fprintf(' dY(SSCB): (%d, %d, %d, %d)\n',size(dY))
fprintf(' w(SSCyCx): (%d, %d, %d, %d)\n',size(kernels_back))
fprintf(' dX(SSCB): (%d, %d, %d, %d)\n',size(dX))
toc

tic
% graident of the loss to the weights
fprintf("\ncpu+dlarray: computing griadent in the convolutional layer... \n")
X=dlarray(x,'SSUC');
dY=dlarray(dy,'SSUC');
DLDW=dlconv(X,dY,0,'Padding',(k-1)/2);
DLDW=permute(stripdims(DLDW),[1,2,4,3]);
fprintf(' DLDW=X*dY:\n')
fprintf(' X(SSCB): (%d, %d, %d, %d)\n',size(X))
fprintf(' dY(SSCB): (%d, %d, %d, %d)\n',size(dY))
fprintf(' DLDW(SSCxCy): (%d, %d, %d, %d)\n',size(DLDW))
toc


%%%%%%%%%% running on CPU %%%%%%%%%%%%%%%
rng('default')
fprintf('############ runing on CPU (valid convolution) ############')
x=randn(height,width,channel_x,batchsize,'single');
dx=zeros(height,width,channel_x,batchsize,'single');
kernels=randn(k,k,channel_x,channel_y,'single');
y=zeros(height-k+1,width-k+1,channel_y,batchsize,'single');
dy=randn(height-k+1,width-k+1,channel_y,batchsize,'single');
DLDw=zeros(size(kernels),'single');

% convolutional layer forwarding
tic
fprintf("\ncpu: forwarding through convolutional layer... \n")
for out = 1:channel_y
    for in = 1:channel_x
        kernel=kernels(:,:,in,out);
        y(:,:,out,:)=y(:,:,out,:)+convn(x(:,:,in,:),rot90(kernel,2),'valid');
    end
end
fprintf(' y=x*w:\n')
fprintf(' x(SSCB): (%d, %d, %d, %d)\n',size(x))
fprintf(' w(SSCxCy): (%d, %d, %d, %d)\n',size(kernels))
fprintf(' y(SSCB): (%d, %d, %d, %d)\n',size(y))
toc

% convolutional layer error backpropagatiing
fprintf("\ncpu: backpropagating through convolutional layer... \n")
tic
for out = 1:channel_y
    for in = 1:channel_x
        kernel=kernels(:,:,in,out);
        dx(:,:,in,:)=dx(:,:,in,:)+convn(dy(:,:,out,:),kernel,'full');
    end
end
fprintf(' dx=dy*wT:\n')
fprintf(' dy(SSCB): (%d, %d, %d, %d)\n',size(dy))
fprintf(' w(SSCxCy): (%d, %d, %d, %d)\n',size(kernels))
fprintf(' dx(SSCB): (%d, %d, %d, %d)\n',size(dx))
toc

% griadent of the loss to the weights
fprintf("\ncpu: computing griadent in the convolutional layer... \n")
tic
for out = 1:channel_y
    for in = 1:channel_x
        for bb=1:batchsize
            DLDw(:,:,in,out)=DLDw(:,:,in,out)+convn(x(:,:,in,bb),rot90(dy(:,:,out,bb),2),'valid');
        end
    end
end
fprintf(' DLDw=x*dy:\n')
fprintf(' x(SSCB): (%d, %d, %d, %d)\n',size(x))
fprintf(' dy(SSCB): (%d, %d, %d, %d)\n',size(dy))
fprintf(' DLDw(SSCxCy): (%d, %d, %d, %d)\n',size(DLDw))
toc

%%
% convolutional layer forwarding
tic
fprintf("\n\ncpu+dlarray: forwarding through convolutional layer... \n")
X=dlarray(x,'SSCB');
Y=dlconv(X,kernels,0);

fprintf(' Y=X*w:\n')
fprintf(' X(SSCB): (%d, %d, %d, %d)\n',size(X))
fprintf(' w(SSCxCy): (%d, %d, %d, %d)\n',size(kernels))
fprintf(' Y(SSCB): (%d, %d, %d, %d)\n',size(Y))
toc

Z=sigmoid(Y);
[P,indx]=maxpool(Z,2,'Stride',2);

% convolutional layer error backpropagatiing
tic
fprintf("\ncpu+dlarray: backpropagating through convolutional layer... \n")
dY=dlarray(dy,'SSCB');
kernels_back=permute(rot90(kernels,2),[1,2,4,3]);
dX=dlconv(dY,kernels_back,0,'Padding',(k-1));

fprintf(' dX=dY*wT:\n')
fprintf(' dY(SSCB): (%d, %d, %d, %d)\n',size(dY))
fprintf(' w(SSCyCx): (%d, %d, %d, %d)\n',size(kernels_back))
fprintf(' dX(SSCB): (%d, %d, %d, %d)\n',size(dX))
toc

% graident of the loss to the weights
fprintf("\ncpu+dlarray: computing griadent in the convolutional layer... \n")
X=dlarray(x,'SSUC');
dY=dlarray(dy,'SSUC');
DLDW=dlconv(X,dY,0);
DLDW=permute(stripdims(DLDW),[1,2,4,3]);
fprintf(' DLDW=X*dY:\n')
fprintf(' X(SSCB): (%d, %d, %d, %d)\n',size(X))
fprintf(' dY(SSCB): (%d, %d, %d, %d)\n',size(dY))
fprintf(' DLDW(SSCxCy): (%d, %d, %d, %d)\n',size(DLDW))
toc