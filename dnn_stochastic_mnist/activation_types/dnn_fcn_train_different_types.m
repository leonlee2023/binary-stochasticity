clc
clear
close all

tic
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

images_tr=gpuarray(images_tr);
images_ts=gpuarray(images_ts);
labels_tr=gpuarray(labels_tr);
labels_ts=gpuarray(labels_ts);

dnn.act_a=1;
act_type='logistic-ReLU';  % TYPE={'logistic','ReLU','Truncated-ReLU','Truncated-ReLU2','logistic-ReLU','ReLU-logistic'};
max_epoch=1000;

save_folder=sprintf('results_%s_ep%i',act_type,max_epoch);
if ~exist(save_folder,'dir')
    mkdir(save_folder);
end

type={'HP','S'}; 
for ff=1:numel(type)
    for ee=1:numel(type)
        for dd=1:numel(type)
            dnn.act_type.type=act_type;
            dnn.act_type.forward='HP';%%type{ff}; 
            dnn.act_type.error='S';%%type{ee};    
            dnn.act_type.derivative='HP';%%type{dd};   
            
            rng('default');
            dnn.batch_size=100;
            dnn.num_batches=size(images_tr,1)/dnn.batch_size;
            dnn.max_epoch=max_epoch;
            dnn.Loss_epoch=zeros(1,dnn.max_epoch);
            dnn.test_accuracy=zeros(1,dnn.max_epoch);
            dnn.train_accuracy=zeros(1,dnn.max_epoch);
            dnn.eta=0.1;
            
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
            
                dnn.nn(ll).weight=gpuarray(sqrt(2/dnn.nn(ll).in_size)*randn(dnn.nn(ll).in_size,dnn.nn(ll).out_size));    
                dnn.nn(ll).bias=gpuarray(zeros(1,dnn.nn(ll).out_size));
                
            
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
            
                        data.nn(ll).y=data.nn(ll).x*dnn.nn(ll).weight+dnn.nn(ll).bias;
                        if ll==dnn.n_layers
                            data.nn(ll).z = exp(data.nn(ll).y)./sum(exp(data.nn(ll).y),2);  
                        else
                            [data.nn(ll).z, data.nn(ll).DzDy] = activation(data.nn(ll).y,dnn.act_type.type,dnn.act_a);  
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
            
                        data.nn(ll).DLDw=(data.nn(ll).x'*data.nn(ll).DLDy)/dnn.batch_size;
                        data.nn(ll).DLDx=data.nn(ll).DLDy*dnn.nn(ll).weight';
            
                        dnn.nn(ll).weight=dnn.nn(ll).weight - dnn.eta*(data.nn(ll).DLDw);
                        dnn.nn(ll).bias=dnn.nn(ll).bias - dnn.eta*sum(data.nn(ll).DLDy,1)/dnn.batch_size;
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

            
            filename=sprintf([save_folder,'/trained_dnn_[%s]_ep%d_eta%.2f_F%s_E%s_D%s_',datestr(datetime,'yymmdd_HHMM')],...
                            dnn.architecture,dnn.max_epoch,dnn.eta,dnn.act_type.forward,dnn.act_type.error,dnn.act_type.derivative);
            save([filename,'.mat'],'dnn') 
            savefig(fig_err_acc,[filename,'_acc.fig']) 
            savefig(fig_wt_dist,[filename,'_wt.fig']) 

        end
    end
end
