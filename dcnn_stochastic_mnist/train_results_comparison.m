clc
clear
close all


load trained_dnn_conv_ep1000_eta0.10_HP_220829_0532.mat
test_accuracy_hp=dnn.accuracy_test;

load trained_dnn_conv_ep1000_eta0.10_stochastic_220831_0837.mat
test_accuracy_s=dnn.accuracy_test;

figure;
semilogy(1:numel(test_accuracy_hp),(1-test_accuracy_hp)*100,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); hold on
semilogy(1:numel(test_accuracy_s),(1-test_accuracy_s)*100,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); 
% semilogy(1:numel(train_accuracy_hp),(1-train_accuracy_hp)*100,'-','linewidth',2,'markersize',12,'markerfacecolor','w'); hold on
% semilogy(1:numel(train_accuracy_s),(1-train_accuracy_s)*100,'-','linewidth',2,'markersize',12,'markerfacecolor','w'); 
xlabel('Epoch');ylabel("Test Error [%]");
% legend({'HP - test','Stochastic - test','HP - train','Stochastic - train'},'location','northeast');
legend({'HP training','Stochastic training'},'location','northeast');
ylim([1,5]);grid on;
set(gca,'fontsize',15,'linewidth',1.5);

max(test_accuracy_hp)
max(test_accuracy_s)

%%
clc
clear
close all

load trained_dnn_conv_ep1000_eta0.10_HP_220829_0532.mat
test_accuracy_hp=dnn.accuracy_test;
train_accuracy_hp=dnn.accuracy_train;
loss_hp=dnn.Loss_epoch;

load trained_dnn_conv_ep1000_eta0.10_stochastic_220831_0837.mat
test_accuracy_s=dnn.accuracy_test;
train_accuracy_s=dnn.accuracy_train;
loss_s=dnn.Loss_epoch;

figure;
ax1=subplot(3,1,1);
semilogy(1:numel(loss_hp),loss_hp,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); hold on
semilogy(1:numel(loss_s),loss_s,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); 
ylabel("Cross Entropy");grid on;xticklabels([]);
yticks(10.^(-5:1:0));
legend({'HP training','Stochastic training'},'location','northeast');
set(gca,'fontsize',15,'linewidth',1.5);
ax1.Position=[0.1300 0.64 0.7750 0.29];

ax2=subplot(3,1,[2,3]);
semilogy(1:numel(test_accuracy_hp),(1-test_accuracy_hp)*100,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); hold on
semilogy(1:numel(test_accuracy_s),(1-test_accuracy_s)*100,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); 
semilogy(1:numel(train_accuracy_hp),(1-train_accuracy_hp)*100,'-','linewidth',2,'markersize',12,'markerfacecolor','w'); hold on
semilogy(1:numel(train_accuracy_s),(1-train_accuracy_s)*100,'-','linewidth',2,'markersize',12,'markerfacecolor','w'); 
xlabel('Epoch');ylabel("Test Error [%]");
legend({'HP training - test set','Stochastic training - test set','HP training - train set','Stochastic training - train set'},'location','northeast');
% legend({'HP training','Stochastic training'},'location','northeast');
ylim([1e-3,20]);grid on;yticks(10.^(-2:1:1));
set(gca,'fontsize',15,'linewidth',1.5);

max(test_accuracy_hp)
max(test_accuracy_s)