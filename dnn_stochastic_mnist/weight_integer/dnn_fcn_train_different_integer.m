clc
clear
close all

tic

WTYPE={'INT8','INT7','INT6','INT5','INT4','INT3','INT2','Ternary'};
TYPE={'HP','S'};

global usegpu;
if gpuDeviceCount>0
    usegpu=true;
else
    usegpu=false;
end

load('../../dataset/mnist/MNIST.mat')
images_tr=reshape(images_tr,[],size(images_tr,3))';
images_ts=reshape(images_ts,[],size(images_ts,3))';
labels_tr=labels_tr';
labels_ts=labels_ts';
save_folder='different_integer/';

images_tr=gpuarray(images_tr);
images_ts=gpuarray(images_ts);
labels_tr=gpuarray(labels_tr);
labels_ts=gpuarray(labels_ts);

for ww=1:numel(WTYPE)
    
    dnn.w_type=WTYPE{ww};             % data type of weight elements
    %%%%% weight limit: [-1,1] %%%%%
    switch dnn.w_type
        case 'INT8'
            dnn.scalor=128;
        case 'INT7'
            dnn.scalor=64; 
        case 'INT6'
            dnn.scalor=32;
        case 'INT5'
            dnn.scalor=16; 
        case 'INT4'
            dnn.scalor=8; 
        case 'INT3'
            dnn.scalor=4; 
        case 'INT2'
            dnn.scalor=2; 
        case 'Ternary'
            dnn.scalor=2; 
    end
    
    for tt=1:numel(TYPE)
        type=TYPE{tt};       % 
        
        dnn.act_type.forward=type;    % activation type for forwad pass output, HP: high precision; S: sampled; B: binary
        dnn.act_type.error=type;      % error type for backward propogation pass, HP: high precision; S: signed 
        dnn.act_type.derivative=type; % type of activation derivative, HP: high precision; S: sampled
        dnn.der_aug=4;               % prefactor "a" for the activation function, z=1/(1+exp(-a*y));
        
        rng('default');
        dnn.batch_size=100;
        dnn.num_batches=size(images_tr,1)/dnn.batch_size;
        dnn.max_epoch=2;
        dnn.eta=0.1;                 % learning rate
        dnn.Loss_epoch=zeros(1,dnn.max_epoch);
        dnn.test_accuracy=zeros(1,dnn.max_epoch);
        dnn.train_accuracy=zeros(1,dnn.max_epoch);
        
        dnn.hidden_size=[500,200];
        dnn.n_layers=numel(dnn.hidden_size)+1;
        
        fprintf('Initializing the DNN ... \n')
        for ll=1:dnn.n_layers
            if ll==1
                dnn.nn(ll).in_size=size(images_tr,2);
                dnn.architecture=num2str(dnn.nn(ll).in_size);
            else
                dnn.nn(ll).in_size=dnn.hidden_size(ll-1);
                dnn.architecture=[dnn.architecture,'-',num2str(dnn.nn(ll).in_size)];
            end
            if ll==dnn.n_layers
                dnn.nn(ll).out_size=size(labels_tr,2);
                dnn.architecture=[dnn.architecture,'-',num2str(dnn.nn(ll).out_size)];
            else
                dnn.nn(ll).out_size=dnn.hidden_size(ll);
            end
        
            if usegpu
                dnn.nn(ll).weight=sqrt(2/dnn.nn(ll).in_size)*randn(dnn.nn(ll).in_size,dnn.nn(ll).out_size,'single','gpuArray');    
                dnn.nn(ll).weight=int8(dnn.nn(ll).weight*dnn.scalor);
                dnn.nn(ll).bias=zeros(1,dnn.nn(ll).out_size,'int8','gpuArray');
                data.nn(ll).DLDw=zeros(size(dnn.nn(ll).weight),'single','gpuArray');    
                data.nn(ll).DLDb=zeros(size(dnn.nn(ll).bias),'single','gpuArray');
            else
                dnn.nn(ll).weight=sqrt(2/dnn.nn(ll).in_size)*randn(dnn.nn(ll).in_size,dnn.nn(ll).out_size,'single');    
                dnn.nn(ll).weight=int8(dnn.nn(ll).weight*dnn.scalor);
                dnn.nn(ll).bias=zeros(1,dnn.nn(ll).out_size,'int8');
                data.nn(ll).DLDw=zeros(size(dnn.nn(ll).weight),'single');    
                data.nn(ll).DLDb=zeros(size(dnn.nn(ll).bias),'single');
            end
            [dnn.nn(ll).weight,data.nn(ll).DLDw] = weight_update(dnn.nn(ll).weight,data.nn(ll).DLDw,dnn);
            [dnn.nn(ll).bias,data.nn(ll).DLDb] = weight_update(dnn.nn(ll).bias,data.nn(ll).DLDb,dnn);

            
            fprintf('  Neural Network %d:\n',ll)
            fprintf('    weight size: [%dx%d]\n', dnn.nn(ll).in_size,dnn.nn(ll).out_size)
        end
        
        dnn.total_weight_number=0;
        for ll=1:dnn.n_layers
            dnn.total_weight_number=dnn.total_weight_number+numel(dnn.nn(ll).weight);
        end
        fprintf('  Total number of weights: %d\n', dnn.total_weight_number)
        
        fprintf('The DNN initialized. \n\n')
        
        for epoch=1:dnn.max_epoch
            t_epoch=tic;
            fprintf('  Epoch: %d, ',epoch)
            Loss_batch=zeros(1,dnn.num_batches);
            for batch=1:dnn.num_batches
                num_imgs=((batch-1)*dnn.batch_size+1):batch*dnn.batch_size;
                input=images_tr(num_imgs,:);
                label=labels_tr(num_imgs,:);
        
                %%%%%%%%%%%%%%%%%%%%%%%% Forward PASS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
                for ll=1:dnn.n_layers
                    if ll==1
                        data.nn(ll).x=sampling(input,dnn.act_type.forward);
                    else
                        data.nn(ll).x=data.nn(ll-1).z;
                    end
        
                    data.nn(ll).y=(data.nn(ll).x*single(dnn.nn(ll).weight)+single(dnn.nn(ll).bias))/dnn.scalor;
                    if ll==dnn.n_layers
                        data.nn(ll).z = exp(data.nn(ll).y)./sum(exp(data.nn(ll).y),2);  
                    else
                        [data.nn(ll).z, data.nn(ll).DzDy] = activation(data.nn(ll).y,dnn.der_aug);  
                        data.nn(ll).z    = sampling(data.nn(ll).z,dnn.act_type.forward);
                        data.nn(ll).DzDy = sampling(data.nn(ll).DzDy,dnn.act_type.derivative);
                    end
                end
        
                %%%%%%%%%%%%%%%%%%%%%%%% ERROR COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%
                Loss_batch(batch) = -sum(label.*log(data.nn(end).z),'all')/dnn.batch_size;          
                
                %%%%%%%%%%%%%%%%%%%%%%%% BACKWARD PASS and weight update %%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
                for ll=dnn.n_layers:-1:1
                    if ll==dnn.n_layers
                        data.nn(ll).z=sampling(data.nn(ll).z,dnn.act_type.forward);
                        if isequal(dnn.act_type.error,'S')
                            data.nn(ll).z=sampling(data.nn(ll).z,'S');
                        end
                        data.nn(ll).DLDy = data.nn(ll).z - label;
                    else
                        data.nn(ll).DLDz=data.nn(ll+1).DLDx;
                        % binarize the error inputs to each layer weight matrix
                        data.nn(ll).DLDz=error_binarize(data.nn(ll).DLDz, dnn.act_type.error);
                        data.nn(ll).DLDy=data.nn(ll).DLDz.*data.nn(ll).DzDy;
                    end     
                    data.nn(ll).DLDx=data.nn(ll).DLDy*single(dnn.nn(ll).weight')/dnn.scalor;
        
                    data.nn(ll).DLDw=data.nn(ll).DLDw+(data.nn(ll).x'*data.nn(ll).DLDy);
                    data.nn(ll).DLDb=data.nn(ll).DLDb+(sum(data.nn(ll).DLDy,1));
        
                    [dnn.nn(ll).weight,data.nn(ll).DLDw] = weight_update(dnn.nn(ll).weight,data.nn(ll).DLDw,dnn);
                    [dnn.nn(ll).bias,data.nn(ll).DLDb] = weight_update(dnn.nn(ll).bias,data.nn(ll).DLDb,dnn);
                end
        
        
            end
            dnn.Loss_epoch(epoch)=mean(Loss_batch);
        
            % test on train and test sets
            dnn.train_accuracy(epoch)=test(images_tr,labels_tr,dnn);
            dnn.test_accuracy(epoch)=test(images_ts,labels_ts,dnn);
        
            fprintf('Loss: %d; Accuracy: %.3f%% (train); %.3f%% (test).\n',...
                dnn.Loss_epoch(epoch),dnn.train_accuracy(epoch)*100,dnn.test_accuracy(epoch)*100);
            toc(t_epoch);
        end
        toc;dnn.total_trian_time=toc;
        
        
        fig_err_acc=figure;
        subplot(2,1,1)
        plot(1:numel(dnn.Loss_epoch),dnn.Loss_epoch,'s-','linewidth',2,'markersize',12,'markerfacecolor','w'); hold on;
        set(gca,'fontsize',15,'linewidth',1.5);
        ylabel("Loss");
        
        subplot(2,1,2)
        plot(1:numel(dnn.train_accuracy),dnn.train_accuracy*100,'o-','linewidth',2,'markersize',12,'markerfacecolor','w'); hold on;
        plot(1:numel(dnn.test_accuracy),dnn.test_accuracy*100,'o-','linewidth',2,'markersize',12,'markerfacecolor','w'); 
        xlabel('Epoch');ylabel("Accuracy [%]");legend({'Train','Test'});
        set(gca,'fontsize',15,'linewidth',1.5);
        
        fig_wt_dist=figure;
        leg_str=[];
        for ll=1:dnn.n_layers
            histogram(dnn.nn(ll).weight,'Normalization','probability'); hold on;
            leg_str{ll}=sprintf('Layer %d',ll); %#ok<SAGROW> 
        end
        set(gca,'fontsize',15,'linewidth',1.5);
        ylabel("Probability");xlabel('Weight')
        legend(leg_str)
        
        for ll=1:dnn.n_layers
            dnn.nn(ll).weight=gather(dnn.nn(ll).weight);
            dnn.nn(ll).bias=gather(dnn.nn(ll).bias);
        end
        
        filename=sprintf([save_folder,'trained_dnn_[%s]_ep%d_eta%.2f_w%s_sc%d_type%s_',datestr(datetime,'yymmdd_HHMM')],...
                    dnn.architecture,dnn.max_epoch,dnn.eta,dnn.w_type,dnn.scalor,type);
        save([filename,'.mat'],'dnn') 
        savefig(fig_err_acc,[filename,'_acc.fig']) 
        savefig(fig_wt_dist,[filename,'_wt.fig']) 
    end
end
