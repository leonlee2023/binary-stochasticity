clc
clear
close all


filename=sprintf('*FHP_EHP_DHP_*.mat');
file = dir(filename);
load(file.name)
test_accuracy_hp=dnn.test_accuracy;
train_accuracy_hp=dnn.train_accuracy;
loss_hp=dnn.Loss_epoch;

filename=sprintf('*FS_ES_DS_*.mat');
file = dir(filename);
load(file.name)
test_accuracy_s=dnn.test_accuracy;
train_accuracy_s=dnn.train_accuracy;
loss_s=dnn.Loss_epoch;

fig=figure;
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
legend({'HP training - test set','Stochastic training - test set','HP training - train set','Stochastic training - train set'},'location','east');
% legend({'HP training','Stochastic training'},'location','northeast');
ylim([0.001,20]);grid on;yticks(10.^(-2:1:1));
set(gca,'fontsize',15,'linewidth',1.5);

max(test_accuracy_hp)
max(test_accuracy_s)

savefig(fig,'test_results_comparision_hp_vs_stochastic.fig')