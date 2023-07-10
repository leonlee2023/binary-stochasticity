clc
clear
close all

tic

load('../dataset/mnist/MNIST.mat')

dnn.type='stochastic'; % 'HP' or 'stochastic'
dnn.act_type.act_a=4;
switch dnn.type
    case 'HP'
        dnn.act_type.forward='HP'; % 'sampling', 'analog', 'binary':activation type for forwad pass output
        dnn.act_type.error='HP';     % 'analog', 'binary' : error type for backward propogation pass
        dnn.act_type.derivative='HP';  % 'sampling', 'analog': 
    case 'stochastic'
        dnn.act_type.forward='S';
        dnn.act_type.error='S';    
        dnn.act_type.derivative='S';  
    otherwise
        error('Neural network type not defined!')
end

% rng('default');
dnn.batch_size=100;
dnn.num_batches=size(images_tr,3)/dnn.batch_size;
dnn.max_epoch=100;
dnn.Loss_epoch=zeros(1,dnn.max_epoch);
dnn.accuracy_test=zeros(1,dnn.max_epoch);
dnn.accuracy_train=zeros(1,dnn.max_epoch);
dnn.eta=0.1;

% defining the parameters of the conv layers
dnn.n_conv=2;          % number of conv layers
dnn.n_kernels=[9,5];   % convolutional kenerl size
dnn.n_pools=[2,2];     % pooling batch size
dnn.n_features=[8,12]; % number of output (hidden/pooling) feature maps
dnn.n_strides=[1,1];   % convolutional stride step size

fprintf('Initializing the Convolutional DNN ... \n')
for layer=1:dnn.n_conv

    conv.act_type.act_a=dnn.act_type.act_a;
    conv.n_kernel=dnn.n_kernels(layer);
    conv.n_stride=dnn.n_strides(layer);
    conv.n_pool=dnn.n_pools(layer);
    conv.output_n_feature=dnn.n_features(layer);
    if layer==1
        conv.input_size=size(images_tr,1);
        conv.input_n_feature=1;
    else
        conv.input_size=dnn.conv(layer-1).pool_size;
        conv.input_n_feature=dnn.conv(layer-1).output_n_feature;
    end
    conv.output_size=((conv.input_size-conv.n_kernel)/conv.n_stride+1);
    conv.pool_size=conv.output_size/conv.n_pool;

    conv.weight=sqrt(2/(conv.n_kernel^2*conv.input_n_feature))*randn(conv.n_kernel,conv.n_kernel,conv.input_n_feature,conv.output_n_feature);
%     conv.weight=0.1*randn(conv.n_kernel,conv.n_kernel,conv.input_n_feature,conv.output_n_feature);
    conv.biases=zeros(conv.output_size,conv.output_size,conv.output_n_feature);

    fprintf('  Conv %d:\n',layer)
    fprintf('    input size: [%dx%dx%d]\n', conv.input_size,conv.input_size,conv.input_n_feature)
    fprintf('    weight size: [%dx%dx%dx%d]\n', conv.n_kernel,conv.n_kernel,conv.input_n_feature,conv.output_n_feature)
    fprintf('    output size: [%dx%dx%d]\n', conv.output_size,conv.output_size,conv.output_n_feature)
    fprintf('    pool size: [%dx%dx%d]\n', conv.pool_size,conv.pool_size,conv.output_n_feature)

    dnn.conv(layer)=conv;
end

full.act_type.act_a=dnn.act_type.act_a;
full.input_size=(dnn.conv(end).pool_size)^2*dnn.conv(end).output_n_feature;
full.output_size=size(labels_tr,1);

full.weight=sqrt(2/full.input_size)*randn(full.input_size,full.output_size);
% full.weight=0.1*randn(full.input_size,full.output_size);
full.biases=zeros(1,full.output_size);


fprintf('  Full:\n')
fprintf('    input size: [%dx%d]\n', 1,full.input_size)
fprintf('    output size: [%dx%d]\n', 1,full.output_size)
fprintf('    weight size: [%dx%d]\n', full.input_size,full.output_size)

dnn.full=full;

dnn.total_weight_number=0;
for layer=1:dnn.n_conv
    dnn.total_weight_number=dnn.total_weight_number+numel(dnn.conv(layer).weight);
end
dnn.total_weight_number=dnn.total_weight_number+numel(dnn.full.weight);
fprintf('\n    Total number of weights: %d\n', dnn.total_weight_number)


time=toc;
fprintf('Training the convolutional DNN ... \n')
fprintf('  Epoch:     Loss:      Accuracy: \n')
fprintf('                       (train) (test).\n')

for epoch=1:dnn.max_epoch
    fprintf('  %d, ',epoch)
    Loss_batch=zeros(1,dnn.num_batches);
    nbits=fprintf('(%.2f%%)',0);
    for batch=1:dnn.num_batches
        
        num_imgs=((batch-1)*dnn.batch_size+1):batch*dnn.batch_size;
        input=reshape(images_tr(:,:,num_imgs),[dnn.conv(1).input_size,dnn.conv(1).input_size,1,dnn.batch_size]);
        data.label=labels_tr(:,num_imgs)';

        %%%%%%%%%%%%%%%%%%%%%%%% Forward PASS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data.conv(1).x=sampling(input,dnn.act_type.forward);
        for ll=1:dnn.n_conv
            [data.conv(ll).y,data.conv(ll).z,data.conv(ll).DzDy,data.conv(ll).p,data.conv(ll).DpDz]...
                =conv_forward(data.conv(ll).x,dnn.conv(ll));
            data.conv(ll).DzDy=sampling(data.conv(ll).DzDy, dnn.act_type.derivative);
            data.conv(ll).p=sampling(data.conv(ll).p,dnn.act_type.forward);
            if ll<dnn.n_conv
                data.conv(ll+1).x=data.conv(ll).p;
            end
        end

        data.full.x=reshape(data.conv(ll).p,[],dnn.batch_size)';
        data.full.y=data.full.x*dnn.full.weight+dnn.full.biases;
        data.full.z = exp(data.full.y)./sum(exp(data.full.y),2); 

        %%%%%%%%%%%%%%%%%%%%%%%% ERROR COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%
        Loss_batch(batch) = -sum(data.label.*log(data.full.z),'all')/dnn.batch_size; 
        
        if toc-time>1  % print the progress of epoach per second
            time=toc;
            fprintf(repmat('\b',1,nbits));
            nbits=fprintf('(%.2f%%)',batch/dnn.num_batches*100);
            nbits=nbits+fprintf(', batch loss: %d',Loss_batch(batch));
        end

        %%%%%%%%%%%%%%%%%%%%%%%% BACKWARD PASS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data.full.z=sampling(data.full.z,dnn.act_type.forward);
        if isequal(dnn.act_type.error,'S')
            data.full.z=sampling(data.full.z,'S');
        end
        data.full.DLDy = data.full.z - data.label;
        data.full.DLDx=data.full.DLDy*dnn.full.weight';

        for ll=dnn.n_conv:-1:1
            if ll==dnn.n_conv
                data.conv(ll).DLDp=reshape(data.full.DLDx',dnn.conv(ll).pool_size,dnn.conv(ll).pool_size,dnn.conv(ll).output_n_feature,dnn.batch_size);
            else
                data.conv(ll).DLDp=data.conv(ll+1).DLDx;
            end
            data.conv(ll).DLDp=error_binarize(data.conv(ll).DLDp, dnn.act_type.error);
            [data.conv(ll).DLDx,data.conv(ll).DLDy,data.conv(ll).DLDz]...
                =conv_backward(data.conv(ll).DLDp,data.conv(ll).DpDz,data.conv(ll).DzDy,dnn.conv(ll));
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%% Weight update %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        data.full.DLDw=data.full.x'*data.full.DLDy/dnn.batch_size;
        data.full.DLDb=sum(data.full.DLDy,1)/dnn.batch_size;
        dnn.full.weight=dnn.full.weight - dnn.eta*(data.full.DLDw);
        dnn.full.biases=dnn.full.biases - dnn.eta*(data.full.DLDb);

        for ll=dnn.n_conv:-1:1
            data.conv(ll).DLDw=conv_gradient(data.conv(ll).x, data.conv(ll).DLDy,dnn.conv(ll));
            data.conv(ll).DLDb=sum(data.conv(ll).DLDy,4)/dnn.batch_size;
            dnn.conv(ll).weight=dnn.conv(ll).weight - dnn.eta*(data.conv(ll).DLDw);
            dnn.conv(ll).biases=dnn.conv(ll).biases - dnn.eta*(data.conv(ll).DLDb);
        end

    end
    dnn.Loss_epoch(epoch)=mean(Loss_batch);
    
    dnn.accuracy_train(epoch)=test(images_tr,labels_tr,dnn);
    dnn.accuracy_test(epoch)=test(images_ts,labels_ts,dnn);

    fprintf(repmat('\b',1,nbits));
    fprintf('(%.2f%%): ',100);

    fprintf('     %d,    %.4f%%, %.4f%%.\n',dnn.Loss_epoch(epoch),...
        dnn.accuracy_train(epoch)*100,dnn.accuracy_test(epoch)*100); 
    toc;
end

fprintf('  ')
toc;dnn.total_time=toc;
fprintf('Convolutional DNN training finished.\n\n')


fig_err_acc=figure;
subplot(2,1,1)
plot(1:numel(dnn.Loss_epoch),dnn.Loss_epoch,'s-','linewidth',2,'markersize',12,'markerfacecolor','w'); hold on;
set(gca,'fontsize',15,'linewidth',1.5);
ylabel("Loss");

subplot(2,1,2)
plot(1:numel(dnn.accuracy_train),dnn.accuracy_train*100,'o-','linewidth',2,'markersize',12,'markerfacecolor','w'); hold on;
plot(1:numel(dnn.accuracy_test),dnn.accuracy_test*100,'o-','linewidth',2,'markersize',12,'markerfacecolor','w'); 
xlabel('Epoch');ylabel("Accuracy [%]");legend({'Train','Test'});
set(gca,'fontsize',15,'linewidth',1.5);


fig_wt_dist=figure;
leg_str=[];
for ll=1:dnn.n_conv
    histogram(dnn.conv(ll).weight,'Normalization','probability'); hold on;
    leg_str{ll}=sprintf('Layer %d',ll); %#ok<SAGROW> 
end
histogram(dnn.full.weight,'Normalization','probability'); hold on;
leg_str{ll+1}=sprintf('Layer %d',ll+1); 
set(gca,'fontsize',15,'linewidth',1.5);
ylabel("Probability");xlabel('Weight')
legend(leg_str)

filename=sprintf(['trained_dnn_conv_ep%d_eta%.2f_%s_',datestr(datetime,'yymmdd_HHMM')],...
    dnn.max_epoch,dnn.eta,dnn.type);
save([filename,'.mat'],'dnn') 
savefig(fig_err_acc,[filename,'_acc.fig']) 
savefig(fig_wt_dist,[filename,'_wt.fig']) 

