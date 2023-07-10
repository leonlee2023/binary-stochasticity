clc
clear
close all

tic

load('../../../dataset/mnist/MNIST.mat')
images_tr=reshape(images_tr,[],size(images_tr,3))';
images_ts=reshape(images_ts,[],size(images_ts,3))';
labels_tr=labels_tr';
labels_ts=labels_ts';

load memristive_device_parameters.mat

for cyc=numel(MEM):-1:1
    rng('default');
    % define parameters for memristor devices
    dnn.mem=MEM(cyc);
    
    figure_msb=synpatic_behaivor(dnn.mem);
    title(dnn.mem.publication);drawnow;
    
    % define parameters for memristor update rules for 
    dnn.mem.DLDw_th=64;
    
    dnn.act_a=4;
    dnn.batch_size=dnn.mem.DLDw_th;
    dnn.num_images=size(images_tr,1);
    dnn.num_batches=ceil(dnn.num_images/dnn.batch_size);
    dnn.max_epoch=100;
    dnn.Loss_epoch=zeros(1,dnn.max_epoch);
    dnn.test_accuracy=zeros(1,dnn.max_epoch);
    dnn.train_accuracy=zeros(1,dnn.max_epoch);
    dnn.weight_lims=[-0.5,0.5];
    
    dnn.mem.G0=(dnn.mem.Gmax-dnn.mem.Gmin)/(dnn.weight_lims(2)-dnn.weight_lims(1));
    dnn.mem.I0=dnn.mem.Vread*dnn.mem.G0;
    
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
    
        dnn.nn(ll).weight=weight_initial(dnn.mem,[dnn.nn(ll).in_size,dnn.nn(ll).out_size]); 
        data.nn(ll).DLDw=zeros([dnn.nn(ll).in_size,dnn.nn(ll).out_size]);
        
        fprintf('  Neural Network %d:\n',ll)
        fprintf('    weight size: [%dx%d]\n', dnn.nn(ll).in_size,dnn.nn(ll).out_size)
    end
    
    dnn.total_weight_number=0;
    dnn.total_memristor_number=0;
    for ll=1:dnn.n_layers
        dnn.total_weight_number=dnn.total_weight_number+(dnn.nn(ll).in_size*dnn.nn(ll).out_size);
        dnn.total_memristor_number=dnn.total_memristor_number+dnn.nn(ll).weight.num_memristors;
    end
    fprintf('  Total number of weights: %d; number of memristors: %d\n', dnn.total_weight_number,dnn.total_memristor_number)
    fprintf('The DNN initialized. \n\n')
    
    
    for epoch=1:dnn.max_epoch
        fprintf('  Epoch: %d, ',epoch)
        Loss_batch=zeros(1,dnn.num_batches);
        for batch=1:dnn.num_batches
            num_imgs=((batch-1)*dnn.batch_size+1):min(batch*dnn.batch_size,dnn.num_images);
            input=images_tr(num_imgs,:);
            label=labels_tr(num_imgs,:);
    
            %%%%%%%%%%%%%%%%%%%%%%%% Forward PASS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for ll=1:dnn.n_layers
                if ll==1
                    data.nn(ll).x=sampling(input,'S');
                else
                    data.nn(ll).x=data.nn(ll-1).z;
                end
                data.nn(ll).y=read_matrix(data.nn(ll).x,dnn.nn(ll).weight,dnn.mem);
                if ll==dnn.n_layers
                    data.nn(ll).z = exp(data.nn(ll).y)./sum(exp(data.nn(ll).y),2);  
                else
                    [data.nn(ll).z, data.nn(ll).DzDy] = activation(data.nn(ll).y,dnn.act_a);  
                    data.nn(ll).z    = sampling(data.nn(ll).z,'S');
                    data.nn(ll).DzDy = sampling(data.nn(ll).DzDy,'S');
                end
            end
    
            %%%%%%%%%%%%%%%%%%%%%%%% ERROR COMPUTATION %%%%%%%%%%%%%%%%%%%%%%%%
            Loss_batch(batch) = -sum(label.*log(data.nn(end).z),'all')/dnn.batch_size;             
            
            %%%%%%%%%%%%%%%%%%%%%%%% BACKWARD PASS and weight update %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
            for ll=dnn.n_layers:-1:1
                if ll==dnn.n_layers
                    data.nn(ll).z=sampling(data.nn(ll).z,'S');
                    data.nn(ll).DLDy = data.nn(ll).z - label;
                else
                    data.nn(ll).DLDz=data.nn(ll+1).DLDx;
                    data.nn(ll).DLDz=sign(data.nn(ll).DLDz);
                    data.nn(ll).DLDy=data.nn(ll).DLDz.*data.nn(ll).DzDy;
                end
                
                data.nn(ll).DLDw=data.nn(ll).DLDw+(data.nn(ll).x'*data.nn(ll).DLDy);
                data.nn(ll).DLDx=read_matrix(data.nn(ll).DLDy,dnn.nn(ll).weight,dnn.mem,'back');
                
                [dnn.nn(ll).weight,data.nn(ll).DLDw] = weight_update(dnn.nn(ll).weight,data.nn(ll).DLDw,dnn.mem);
            end
    
    
        end
        dnn.Loss_epoch(epoch)=mean(Loss_batch);
    
        % test on train and test sets
        dnn.train_accuracy(epoch)=test(images_tr,labels_tr,dnn);
        dnn.test_accuracy(epoch)=test(images_ts,labels_ts,dnn);
    
        fprintf('Loss: %d; Accuracy: %.3f%% (train); %.3f%% (test).\n',...
            dnn.Loss_epoch(epoch),dnn.train_accuracy(epoch)*100,dnn.test_accuracy(epoch)*100);
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
    
    
    filename=sprintf(['results/trained_dnn_[%s]_ep%d_%s_',datestr(datetime,'yymmdd_HHMM')],...
        dnn.architecture,dnn.max_epoch,dnn.mem.name);
    save([filename,'.mat'],'dnn')
    savefig(fig_err_acc,[filename,'_acc.fig']) 

end

