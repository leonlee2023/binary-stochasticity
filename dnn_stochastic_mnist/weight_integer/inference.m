clc
clear
close all

load MNIST.mat
% load trained_dnn_[784-500-200-10]_ep100_eta0.10_FHP_EHP_DHP_Daug4_220826_1209.mat
load trained_dnn_[784-500-200-10]_ep100_eta0.10_FS_ES_DS_Daug4_220826_1226.mat

tic
test_acc_analog=test(images_ts,labels_ts,dnn,'HP');
fprintf('Analog forwarding: %.4f%%\n',test_acc_analog*100)
toc

test_acc_binary=test(images_ts,labels_ts,dnn,'B');
fprintf('Binary forwarding: %.4f%%\n',test_acc_binary*100)
toc

test_acc_sampling=zeros(1,20);

for ss=1:20
    test_acc_sampling(ss)=test(images_ts,labels_ts,dnn,'S',ss);
    fprintf('Sampling forwarding (repeat %d times): %.4f%%\n',ss,test_acc_sampling(ss)*100)
    toc
end


close all
figure;
plot(1:numel(test_acc_sampling),test_acc_analog*ones(1,numel(test_acc_sampling))*100,'r--','linewidth',2,'markersize',12,'markerfacecolor','w');hold on 
plot(1:numel(test_acc_sampling),test_acc_binary*ones(1,numel(test_acc_sampling))*100,'b-.','linewidth',2,'markersize',12,'markerfacecolor','w');hold on 

plot(1:numel(test_acc_sampling),test_acc_sampling*100,'go-','linewidth',2,'markersize',12,'markerfacecolor','w'); 
xlabel('Number of sampling');ylabel("Accuracy [%]");
ylim([80,100])
set(gca,'fontsize',15,'linewidth',1.5);
legend({'HP','Binary','Stochastic'},'location','southeast');




