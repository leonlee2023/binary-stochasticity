clc
clear
close all

type={'HP','S'}; % 'sampling', 'analog', 'binary':activation type for forwad pass output

load MNIST.mat

syms={'s','o','^','<','>','d','s','o','^','<','>','d'};
leg_str={};
test_accuracy_all=[];
n=0;
for ff=1:numel(type)
    for ee=1:numel(type)
        for dd=1:numel(type)
            n=n+1;

            rng('default');
            filename=sprintf('results_A4D4_ep1000/*F%s_E%s_D%s_*.mat',type{ff},type{ee},type{dd});
            file = dir(filename);
            load(['results_A4D4_ep1000/',file.name])
            leg_str{n}=sprintf('F:%s,E:%s,D:%s',type{ff},type{ee},type{dd});
            
            test_acc_analog=test(images_ts,labels_ts,dnn,'HP');
            fprintf('Inference 1: %.2f%%\n',(1-test_acc_analog)*100)
            
            test_acc_binary=test(images_ts,labels_ts,dnn,'B');
            fprintf('Inference 2: %.2f%%\n',(1-test_acc_binary)*100)

            
            test_acc_sampling=test(images_ts,labels_ts,dnn,'S',100);
            
            fprintf('Inference 3: %.2f%%-%.2f%%\n',(1-min(test_acc_sampling))*100,(1-max(test_acc_sampling))*100)


            fig=figure;
            semilogy(1:numel(test_acc_sampling),(1-test_acc_analog)*ones(1,numel(test_acc_sampling))*100,'r--','linewidth',2,'markersize',12,'markerfacecolor','w');hold on 
            plot(1:numel(test_acc_sampling),(1-test_acc_binary)*ones(1,numel(test_acc_sampling))*100,'b-.','linewidth',2,'markersize',12,'markerfacecolor','w');hold on 
            
            plot(1:numel(test_acc_sampling),(1-test_acc_sampling)*100,'s-','linewidth',2,'markersize',12,'markerfacecolor','w'); 
            xlabel('Inference repetition');ylabel("Inference Error [%]");
            ylim([1,15]);yticks([1:10,15]);grid on;
            set(gca,'fontsize',15,'linewidth',1.5);
            title(leg_str{n});
            legend({'HP inference','Binary inference','Stochastic inference'},'location','northeast');
            drawnow;
            filename=['results_A4D4_ep1000/',file.name(1:end-4),'_infer.fig'];
            savefig(fig,filename) 

            test_accuracy_all=[test_accuracy_all;[test_acc_analog,test_acc_binary,test_acc_sampling]];
            fprintf('\n')
        end
    end
end

save test_accuracy_all.mat test_accuracy_all leg_str
%%
load test_accuracy_all.mat
fig2=figure;
fig2.Position=[957 447 903 419];
cat = categorical(leg_str);
cat = reordercats(cat,leg_str);
% bar(cat,test_accuracy_all(:,[1,2,3,27,52,102])*100)
% ylim([80,100]);grid on;
bar(cat,(1-test_accuracy_all(:,[1,2,3,27,52,102]))*100);
ylim([1,20]);grid on;
set(gca,'YScale','log')
ylabel("Test Accuracy [%]");
set(gca,'fontsize',15,'linewidth',1.5);
legend({'HP inference','Binary inference','Stochastic Inference #1',...
    'Stochastic Inference #25','Stochastic Inference #50','Stochastic Inference #100'})

