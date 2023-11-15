clc
clear
close all

load MNIST.mat
filename='trained_dnn_[784-500-200-10]_ep100';
load([filename,'.mat'])

tic
test_acc_analog=test(images_ts,labels_ts,dnn,'HP');
fprintf('Analog forwarding: %.4f%%\n',test_acc_analog*100)
toc

test_acc_binary=test(images_ts,labels_ts,dnn,'B');
fprintf('Binary forwarding: %.4f%%\n',test_acc_binary*100)
toc

test_acc_sampling=test(images_ts,labels_ts,dnn,'S',100);

toc

fig_inf=figure;
semilogy(1:numel(test_acc_sampling),(1-test_acc_analog)*ones(1,numel(test_acc_sampling))*100,'r--','linewidth',2,'markersize',12,'markerfacecolor','w');hold on 
plot(1:numel(test_acc_sampling),(1-test_acc_binary)*ones(1,numel(test_acc_sampling))*100,'b-.','linewidth',2,'markersize',12,'markerfacecolor','w');hold on 

plot(1:numel(test_acc_sampling),(1-test_acc_sampling)*100,'s-','linewidth',2,'markersize',12,'markerfacecolor','w'); 
xlabel('Inference repetition');ylabel("Error [%]");
ylim([1,15]);grid on;
set(gca,'fontsize',15,'linewidth',1.5);
legend({'HP inference','Binary inference','Stochastic inference'},'location','northeast');

dnn.test_acc_analog=test_acc_analog;
dnn.test_acc_binary=test_acc_binary;
dnn.test_acc_sampling=test_acc_sampling;

save([filename,'.mat'],'dnn') 
savefig(fig_inf,[filename,'_inference.fig']) 



