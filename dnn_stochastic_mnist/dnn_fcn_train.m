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

fprintf('Loading data and set neural network parameters ...\n')
load('../dataset/mnist/MNIST.mat')
save_folder='';

images_tr=gpuarray(images_tr);
images_ts=gpuarray(images_ts);
labels_tr=gpuarray(labels_tr);
labels_ts=gpuarray(labels_ts);
images_tr=reshape(images_tr,[],size(images_tr,3))';
images_ts=reshape(images_ts,[],size(images_ts,3))';
labels_tr=labels_tr';
labels_ts=labels_ts';

dnn.act_type.forward='S';    % activation type for forwad pass output, HP: high precision (FP32); S: sampled; B: binary
dnn.act_type.error='S';      % error type for backward propogation pass, HP: high precision; S: signed 
dnn.act_type.derivative='S'; % type of activation derivative, HP: high precision; S: sampled
dnn.act_a=4;                 % prefactor "a" for the activation function, z=1/(1+exp(-a*y));

rng('default');
dnn.batch_size=100;
dnn.num_batches=size(images_tr,1)/dnn.batch_size;
dnn.max_epoch=1000;
dnn.Loss_epoch=zeros(1,dnn.max_epoch);
dnn.test_accuracy=zeros(1,dnn.max_epoch);
dnn.train_accuracy=zeros(1,dnn.max_epoch);
dnn.eta=0.1;

dnn.hidden_size=[500,200];     % hidden layer sizes. The input layer size (784) and the output layer size (10) is defined by the dataset.
                               % for instance, hidden_size=[500,200] means that the neural network
                               % have three layers with the size of 784-500-200-10
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

    dnn.nn(ll).weight=gpuarray(sqrt(2/dnn.nn(ll).in_size)*randn(dnn.nn(ll).in_size,dnn.nn(ll).out_size,'single'));    
    dnn.nn(ll).bias=gpuarray(zeros(1,dnn.nn(ll).out_size,'single'));
    
    fprintf('  Neural Network %d:\n',ll)
    fprintf('    weight size: [%dx%d]\n', dnn.nn(ll).in_size,dnn.nn(ll).out_size)
end

dnn.total_weight_number=0;
for ll=1:dnn.n_layers
    dnn.total_weight_number=dnn.total_weight_number+numel(dnn.nn(ll).weight);
end
fprintf('  Total number of weights: %d\n', dnn.total_weight_number)
fprintf('The DNN initialized. \n\n')

fprintf('Start training the neural network ...\n\n')
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
                [data.nn(ll).z, data.nn(ll).DzDy] = activation(data.nn(ll).y,dnn.act_a);  
                data.nn(ll).z    = sampling(data.nn(ll).z,dnn.act_type.forward);
                data.nn(ll).DzDy = sampling(data.nn(ll).DzDy,dnn.act_type.derivative);
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%% Cross Entropy %%%%%%%%%%%%%%%%%%%%%%%%
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
                % binarize the error inputs to each layer weight matrix
                data.nn(ll).DLDz=error_binarize(data.nn(ll+1).DLDx, dnn.act_type.error);
                data.nn(ll).DLDy=data.nn(ll).DLDz.*data.nn(ll).DzDy;
            end     
            data.nn(ll).DLDx=data.nn(ll).DLDy*dnn.nn(ll).weight';

            data.nn(ll).DLDw=(data.nn(ll).x'*data.nn(ll).DLDy);
            data.nn(ll).DLDb=sum(data.nn(ll).DLDy,1);

            dnn.nn(ll).weight=dnn.nn(ll).weight - dnn.eta*data.nn(ll).DLDw/dnn.batch_size;
            dnn.nn(ll).bias=dnn.nn(ll).bias - dnn.eta*data.nn(ll).DLDb/dnn.batch_size;
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
fprintf('Network Training Finished.\n\n')

fprintf('Inference on trained neural network...\n')
rng('default');
t_inf=tic;
test_acc_analog_hp=test(images_ts,labels_ts,dnn,'HP');
fprintf('  Analog forwarding: %.4f%%\n',test_acc_analog_hp*100)
toc(t_inf)

test_acc_binary_hp=test(images_ts,labels_ts,dnn,'B');
fprintf('  Binary forwarding: %.4f%%\n',test_acc_binary_hp*100)
toc(t_inf)

test_acc_sampling_hp=test(images_ts,labels_ts,dnn,'S',100);
toc(t_inf)

dnn.test_acc_analog=test_acc_analog_hp;
dnn.test_acc_binary=test_acc_binary_hp;
dnn.test_acc_sampling=test_acc_sampling_hp;

fprintf('Inference finished.\n\n')

fprintf('Plotting and saving data...\n')
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

fig_inf=figure;
semilogy(1:numel(test_acc_sampling_hp),(1-test_acc_analog_hp)*ones(1,numel(test_acc_sampling_hp))*100,'--','linewidth',2,'markersize',12,'markerfacecolor','w');hold on 
plot(1:numel(test_acc_sampling_hp),(1-test_acc_binary_hp)*ones(1,numel(test_acc_sampling_hp))*100,'-.','linewidth',2,'markersize',12,'markerfacecolor','w');hold on 

plot(1:numel(test_acc_sampling_hp),(1-test_acc_sampling_hp)*100,'s-','linewidth',2,'markersize',12,'markerfacecolor','w'); 
xlabel('Inference repetition');ylabel("Inference Error [%]");
ylim([1,15]);grid on;
set(gca,'fontsize',15,'linewidth',1.5);
legend({'HP inference','Binary inference','Stochastic inference'},'location','northeast');

filename=sprintf([save_folder,'trained_dnn_[%s]_ep%d_eta%.2f_F%s_E%s_D%s_A%d_',datestr(datetime,'yymmdd_HHMM')],...
                dnn.architecture,dnn.max_epoch,dnn.eta,dnn.act_type.forward,dnn.act_type.error,dnn.act_type.derivative,dnn.act_a);
savefig(fig_err_acc,[filename,'_acc.fig']) 
savefig(fig_wt_dist,[filename,'_wt.fig']) 
savefig(fig_inf,[filename,'_inference.fig']) 

for ll=1:dnn.n_layers
    dnn.nn(ll).weight=gather(dnn.nn(ll).weight);
    dnn.nn(ll).bias=gather(dnn.nn(ll).bias);
end

save([filename,'.mat'],'dnn') 
fprintf('Data and figures saved.\n\n')
toc
