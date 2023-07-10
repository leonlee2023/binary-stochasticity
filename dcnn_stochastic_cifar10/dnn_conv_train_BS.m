clc
clear
close all

global usegpu;
if gpuDeviceCount>0
    usegpu=true;
else
    usegpu=false;
end

load('../dataset/cifar10/cifar10.mat')
images_tr=single(images_tr)/255;
images_ts=single(images_ts)/255;

images_tr=gpuarray(images_tr);
images_ts=gpuarray(images_ts);
labels_tr=gpuarray(labels_tr);
labels_ts=gpuarray(labels_ts);
images_tr=dlarray(single(images_tr),'SSCB');  
images_ts=dlarray(single(images_ts),'SSCB');  

load trained_dnn_conv_ep100_eta0.01_HP_221202_1843.mat

tic
dnn.type='S';
dnn.act_type.act_a=4;
switch dnn.type
    case 'HP'
        dnn.act_type.forward='HP'; % 'sampling', 'analog', 'binary':activation type for forwad pass output
        dnn.act_type.error='HP';     % 'analog', 'binary' : error type for backward propogation pass
        dnn.act_type.derivative='HP';  % 'sampling', 'analog': 
    case 'S'
        dnn.act_type.forward='S';
        dnn.act_type.error='S';    
        dnn.act_type.derivative='S';  
    otherwise
        error('Neural network type not defined!')
end

rng('default');
dnn.batch_size=100;
dnn.num_images=size(images_tr,4);
dnn.num_batches=dnn.num_images/dnn.batch_size;
dnn.max_epoch=1000;
dnn.Loss_epoch=zeros(1,dnn.max_epoch);
dnn.accuracy_test=zeros(1,dnn.max_epoch);
dnn.accuracy_train=zeros(1,dnn.max_epoch);

dnn.eta=0.01;
time=toc;
fprintf('Training the convolutional DNN ... \n')
fprintf('  Epoch:     Loss:      Accuracy: \n')
fprintf('                       (train) (test).\n')

for epoch=1:dnn.max_epoch
    fprintf('  %d, ',epoch)
    Loss_batch=zeros(1,dnn.num_batches);
    nbits=fprintf('(%.2f%%)',0);
    correct_count=0;
    for batch=1:dnn.num_batches
        
        num_imgs=((batch-1)*dnn.batch_size+1):batch*dnn.batch_size;
        input=images_tr(:,:,:,num_imgs);
        data.label=labels_tr(:,num_imgs)';

        %%%%%%%%%%%%%%%%%%%%%%%% Forward PASS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             data.conv(1).x=sampling(input,dnn.act_type.forward);
        data.conv(1).x=sampling(input,'HP');
        for ll=1:dnn.n_conv
            [data.conv(ll).y,data.conv(ll).z,data.conv(ll).DzDy,data.conv(ll).p,data.conv(ll).p_idx]...
                =conv_forward(data.conv(ll).x,dnn.conv(ll));
            data.conv(ll).DzDy=sampling(data.conv(ll).DzDy, dnn.act_type.derivative);
            data.conv(ll).p=sampling(data.conv(ll).p,dnn.act_type.forward);
            if ll<dnn.n_conv
                data.conv(ll+1).x=data.conv(ll).p;
            end
        end

        data.conv(1).x=sampling(input,dnn.act_type.forward);

        for ll=1:dnn.n_full
            if ll==1
                data.full(ll).x=reshape(data.conv(dnn.n_conv).p,[],dnn.batch_size)';
            else
                data.full(ll).x=data.full(ll-1).z;
            end
            data.full(ll).y=data.full(ll).x*dnn.full(ll).weight+dnn.full(ll).biases;
            if ll==dnn.n_full
                data.full(ll).z = exp(data.full(ll).y)./sum(exp(data.full(ll).y),2);  
            else
                [data.full(ll).z, data.full(ll).DzDy] = activation(data.full(ll).y,dnn.full(ll).act_a); 
                data.full(ll).z=sampling(data.full(ll).z, dnn.act_type.forward);
                data.full(ll).DzDy=sampling(data.full(ll).DzDy, dnn.act_type.derivative);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%% ERROR COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%
        Loss_batch(batch) = -sum(data.label.*log(data.full(end).z),'all')/dnn.batch_size; 
        
        [~,class] = max(data.full(end).z,[],2);
        [~,target]= max(data.label,[],2);
        correct_count=correct_count+sum(class==target);

        if toc-time>1  % print the progress of epoach per second
            time=toc;
            fprintf(repmat('\b',1,nbits));
            nbits=fprintf('(%.2f%%)',batch/dnn.num_batches*100);
            nbits=nbits+fprintf(', batch loss: %d; batch accuracy: %.2f%%',...
                Loss_batch(batch),sum(class==target)/dnn.batch_size*100);
        end

        %%%%%%%%%%%%%%%%%%%%%%%% BACKWARD PASS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ll=dnn.n_full:-1:1
            if ll==dnn.n_full
                data.full(ll).z=sampling(data.full(ll).z,dnn.act_type.forward);
                data.full(ll).DLDy = data.full(ll).z - data.label;
            else
                data.full(ll).DLDz=data.full(ll+1).DLDx;
                % binarize the error inputs to each layer weight matrix
                data.full(ll).DLDz=error_binarize(data.full(ll).DLDz, dnn.act_type.error);
                data.full(ll).DLDy=data.full(ll).DLDz.*data.full(ll).DzDy;
            end     
            data.full(ll).DLDx=data.full(ll).DLDy*dnn.full(ll).weight';
        end

        for ll=dnn.n_conv:-1:1
            if ll==dnn.n_conv
                data.conv(ll).DLDp=reshape(data.full(1).DLDx',dnn.conv(ll).pool_size,dnn.conv(ll).pool_size,dnn.conv(ll).output_n_feature,dnn.batch_size);
            else
                data.conv(ll).DLDp=data.conv(ll+1).DLDx;
            end
            data.conv(ll).DLDp=error_binarize(data.conv(ll).DLDp, dnn.act_type.error);
            [data.conv(ll).DLDx,data.conv(ll).DLDy,data.conv(ll).DLDz]...
                =conv_backward(data.conv(ll).DLDp,data.conv(ll).p_idx,data.conv(ll).DzDy,dnn.conv(ll));
            
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%% Weight update %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for ll=dnn.n_full:-1:1
            data.full(ll).DLDw=(data.full(ll).x'*data.full(ll).DLDy)/dnn.batch_size;
            data.full(ll).DLDb=sum(data.full(ll).DLDy,1)/dnn.batch_size;

            dnn.full(ll).weight=dnn.full(ll).weight - dnn.eta*(data.full(ll).DLDw);
            dnn.full(ll).biases=dnn.full(ll).biases - dnn.eta*(data.full(ll).DLDb);
        end

        for ll=dnn.n_conv:-1:1
            data.conv(ll).DLDw=conv_gradient(data.conv(ll).x, data.conv(ll).DLDy,dnn.conv(ll))/dnn.batch_size;
            data.conv(ll).DLDb=reshape(stripdims(sum(data.conv(ll).DLDy,[1,2,4])/dnn.batch_size),size(dnn.conv(ll).biases));
            dnn.conv(ll).weight=dnn.conv(ll).weight - dnn.eta*((data.conv(ll).DLDw));
            dnn.conv(ll).biases=dnn.conv(ll).biases - dnn.eta*((data.conv(ll).DLDb));
        end

    end
    dnn.Loss_epoch(epoch)=mean(Loss_batch);
    
    dnn.accuracy_train(epoch)=correct_count/dnn.num_images;
    dnn.accuracy_test(epoch)=test(images_ts,labels_ts,dnn);

    fprintf(repmat('\b',1,nbits));
    fprintf('(%.2f%%): ',100);

    fprintf('     %d,    %.4f%%, %.4f%%.         \n',dnn.Loss_epoch(epoch),...
        dnn.accuracy_train(epoch)*100,dnn.accuracy_test(epoch)*100); 
    toc;
end

fprintf('  ')
toc;dnn.total_time=toc;
fprintf('Convolutional DNN training finished.\n\n')

for ll=1:dnn.n_conv
    dnn.conv(ll).weight=gather(extractdata(dnn.conv(ll).weight));
    dnn.conv(ll).biases=gather(extractdata(dnn.conv(ll).biases));
end

for ll=1:dnn.n_full
    dnn.full(ll).weight=gather(extractdata(dnn.full(ll).weight));
    dnn.full(ll).biases=gather(extractdata(dnn.full(ll).biases));
end


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
leg_str=cell(1,dnn.n_conv+dnn.n_full);
for ll=1:dnn.n_conv
    histogram(dnn.conv(ll).weight,'Normalization','probability'); hold on;
    leg_str{ll}=sprintf('Conv %d',ll); 
end
for ll=1:dnn.n_full
    histogram(dnn.full(ll).weight,'Normalization','probability'); hold on;
    leg_str{dnn.n_conv+ll}=sprintf('Full %d',ll); 
end
set(gca,'fontsize',15,'linewidth',1.5);
ylabel("Probability");xlabel('Weight')
legend(leg_str)

filename=sprintf(['trained_dnn_conv_ep%d_eta%.2f_%s_',datestr(datetime,'yymmdd_HHMM')],...
    dnn.max_epoch,dnn.eta,dnn.type);
save([filename,'.mat'],'dnn') 
savefig(fig_err_acc,[filename,'_acc.fig']) 
savefig(fig_wt_dist,[filename,'_wt.fig']) 

