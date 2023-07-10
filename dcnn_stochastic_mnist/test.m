function accuracy=test(images,labels,dnn,act_type,n_sampling)
    if ~exist('act_type','var')
        act_type='HP';
    end
    if ~exist('n_sampling','var')
        n_sampling=1;
    end

    num_images=size(images,3);

    input=reshape(images,[dnn.conv(1).input_size,dnn.conv(1).input_size,1,num_images]);
    labels=labels';
            
    accuracy=zeros(1,n_sampling);
    z_sum=0;
    for ss=1:n_sampling
        x=sampling(input,act_type);
        for ll=1:dnn.n_conv
            [~,~,~,p,~]=conv_forward(x,dnn.conv(ll));
            p=sampling(p,act_type);
            if ll<dnn.n_conv
                x=p;
            end
        end
    
        x=reshape(p,[],num_images)';
        y=x*dnn.full.weight+dnn.full.biases;
        z = exp(y)./(sum(exp(y),2)); 
        z=sampling(z,act_type);
        
        z_sum=z_sum+z;
        [~,class] = max(z_sum,[],2);
        [~,target]= max(labels,[],2);
        
        accuracy(ss)=sum(class==target)/num_images;
        if n_sampling>1
            fprintf('Sampling forwarding (repeat %d times): %.4f%%\n',ss,accuracy(ss)*100)
        end
    end
    

end