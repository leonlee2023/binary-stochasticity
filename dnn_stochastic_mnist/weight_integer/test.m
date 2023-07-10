function accuracy=test(images,labels,dnn,samp_type,n_sampling)
    if ~exist('samp_type','var')
        samp_type='HP';
    end
    if ~exist('n_sampling','var')
        n_sampling=1;
    end
    
    z_sum=0;
    for ss=1:n_sampling
        for ll=1:dnn.n_layers
            if ll==1
                x=sampling(images,samp_type);
            else
                x=z;
            end
            y=(x*single(dnn.nn(ll).weight)+single(dnn.nn(ll).bias))/dnn.scalor;
            if ll==dnn.n_layers
                z = exp(y)./sum(exp(y),2);
            else
                [z, ~] = activation(y,dnn.der_aug);
            end
            z = sampling(z,samp_type);
        end
        z_sum=z_sum+z;
    end
    [~,class]= max(z_sum,[],2);
    [~,target]=max(labels,[],2);
    accuracy=sum(class==target)/size(images,1);
end