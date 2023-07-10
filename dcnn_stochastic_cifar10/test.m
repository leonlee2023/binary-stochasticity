function accuracy=test(images,labels,dnn,act_type,n_sampling)
    if ~exist('act_type','var')
        act_type='HP';
    end
    if ~exist('n_sampling','var')
        n_sampling=1;
    end

    num_images=size(images,4);
    batch_size=100;
    num_batches=num_images/batch_size;
    labels=labels';

    accuracy=zeros(1,n_sampling);
    z_sum=zeros(size(labels));
    for ss=1:n_sampling
        for batch=1:num_batches
                
            num_imgs=((batch-1)*batch_size+1):batch*batch_size;
            input=images(:,:,:,num_imgs);
    
            x=sampling(input,'HP');
            for ll=1:dnn.n_conv
                [~,~,~,p,~]=conv_forward(x,dnn.conv(ll));
                p=sampling(p,act_type);
                if ll<dnn.n_conv
                    x=p;
                end
            end
        
            for ll=1:dnn.n_full
                if ll==1
                    x=reshape(p,[],batch_size)';
                else
                    x=z;
                end
                y=x*dnn.full(ll).weight+dnn.full(ll).biases;
                if ll==dnn.n_full
                    z = exp(y)./sum(exp(y),2);
                else
                    [z, ~] = activation(y,dnn.full(ll).act_a);
                end
                z=sampling(z,act_type);
            end
            z_sum(num_imgs,:)=z_sum(num_imgs,:)+z;
        end
        
        [~,class] = max(z_sum,[],2);
        [~,target]= max(labels,[],2);
        
        accuracy(ss)=sum(class==target)/num_images;
        if n_sampling>1
            fprintf('Sampling forwarding (repeat %d times): %.4f%%\n',ss,accuracy(ss)*100)
        end
    end
    

end