function [y,z,DzDy,p,DpDz]=conv_forward(x,conv)
% forward pass for convolution and pooling layer

batch_size = size(x, 4);

y=zeros(conv.output_size,conv.output_size,conv.output_n_feature,batch_size);

for inf=1:conv.input_n_feature
    for outf=1:conv.output_n_feature
        weight_kernel=conv.weight(:,:,inf,outf);
        y(:,:,outf,:)=y(:,:,outf,:)+convn(x(:,:,inf,:),rot90(weight_kernel,2),'valid');
    end
end
y=y+conv.biases;
    
[z,DzDy]= activation(y,conv.act_type.act_a); 

% p=zeros(conv.pool_size,conv.pool_size,conv.output_n_feature,batch_size);
% for i=1:conv.n_pool
%     for j=1:conv.n_pool
%         p=p+z(i:conv.n_pool:end,j:conv.n_pool:end,:,:);
%     end
% end
% p=p/(conv.n_pool^2);

DpDz=zeros(size(z));
p=zeros(conv.pool_size,conv.pool_size,conv.output_n_feature,batch_size);

for i=1:conv.pool_size
    for j=1:conv.pool_size
        index_i=((i-1)*conv.n_pool+1):i*conv.n_pool;
        index_j=((j-1)*conv.n_pool+1):j*conv.n_pool;
        z_patch=z(index_i,index_j,:,:);
        [p(i,j,:,:),index]=max(z_patch,[],[1,2],'linear');
        DpDz_patch=zeros(size(z_patch));
        DpDz_patch(index)=1;
        DpDz(index_i,index_j,:,:)=DpDz_patch;
    end
end



