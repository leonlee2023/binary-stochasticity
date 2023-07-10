function [DLDx,DLDy,DLDz]=conv_backward(DLDp,DpDz,DzDy,conv)
% backward pass for convolution layer

batch_size = size(DLDp, 4);

% DLDz=zeros(conv.output_size,conv.output_size,conv.output_n_feature,batch_size);
% 
% for i=1:conv.n_pool
%     for j=1:conv.n_pool
%         DLDz(i:conv.n_pool:end,j:conv.n_pool:end,:,:)=DLDp;%DLDp/(conv.n_pool^2);
%     end
% end

DLDz=zeros(conv.output_size,conv.output_size,conv.output_n_feature,batch_size);

for i=1:conv.pool_size
    for j=1:conv.pool_size
        for index_i=((i-1)*conv.n_pool+1):i*conv.n_pool
            for index_j=((j-1)*conv.n_pool+1):j*conv.n_pool
                DLDz(index_i,index_j,:,:)=DLDp(i,j,:,:);
            end
        end
    end
end
DLDz=DLDz.*DpDz;

DLDy=DLDz.*DzDy;

DLDx=zeros(conv.input_size,conv.input_size,conv.input_n_feature,batch_size);

for inf=1:conv.input_n_feature
    for outf=1:conv.output_n_feature
        weight_kernel=conv.weight(:,:,inf,outf);
        DLDx(:,:,inf,:)=DLDx(:,:,inf,:)+convn(DLDy(:,:,outf,:),weight_kernel,'full');
    end
end


