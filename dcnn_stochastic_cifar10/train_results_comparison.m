clc
clear
close all


load trained_dnn_conv6_full3_ep1000_eta0.01_HP_221208_1231.mat
test_accuracy_hp=dnn.accuracy_test;
figure;
leg_str=cell(1,dnn.n_conv);
for ll=1:dnn.n_conv
    weight=reshape(dnn.conv(ll).weight,1,numel(dnn.conv(ll).weight));
    [f,xi]=ksdensity(weight);
    plot(xi,f,'linewidth',3);hold on;
    leg_str{ll}=['w_{conv',num2str(ll),'}'];
end
set(gca,'fontsize',20,'linewidth',2);             
ylabel("Probability Density");
xlabel('Weight');
legend(leg_str);

load trained_dnn_conv_ep1000_eta0.01_S_221209_0953.mat
test_accuracy_s=dnn.accuracy_test;
figure;
leg_str=cell(1,dnn.n_conv);
for ll=1:dnn.n_conv
    weight=reshape(dnn.conv(ll).weight,1,numel(dnn.conv(ll).weight));
    [f,xi]=ksdensity(weight);
    plot(xi,f,'linewidth',3);hold on;
    leg_str{ll}=['w_{conv',num2str(ll),'}'];
end
set(gca,'fontsize',20,'linewidth',2);             
ylabel("Probability Density");
xlabel('Weight');
legend(leg_str);

figure;
semilogy(1:numel(test_accuracy_hp),(1-test_accuracy_hp)*100,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); hold on
semilogy(1:numel(test_accuracy_s),(1-test_accuracy_s)*100,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); 
% semilogy(1:numel(train_accuracy_hp),(1-train_accuracy_hp)*100,'-','linewidth',2,'markersize',12,'markerfacecolor','w'); hold on
% semilogy(1:numel(train_accuracy_s),(1-train_accuracy_s)*100,'-','linewidth',2,'markersize',12,'markerfacecolor','w'); 
xlabel('Epoch');ylabel("Test Error [%]");
% legend({'HP - test','Stochastic - test','HP - train','Stochastic - train'},'location','northeast');
legend({'HP training','Stochastic training'},'location','northeast');
ylim([10,100]);grid on;
set(gca,'fontsize',15,'linewidth',1.5);

max(test_accuracy_hp)
max(test_accuracy_s)

%%
clc
clear
close all


load trained_dnn_conv_ep100_eta0.01_HP_221202_1843.mat
test_accuracy_hp=dnn.accuracy_test;
train_accuracy_hp=dnn.accuracy_train;
loss_hp=dnn.Loss_epoch;

load trained_dnn_conv_ep1000_eta0.01_S_221206_1914.mat
test_accuracy_s=dnn.accuracy_test;
train_accuracy_s=dnn.accuracy_train;
loss_s=dnn.Loss_epoch;

figure;
ax1=subplot(3,1,1);
semilogy(1:numel(loss_hp),loss_hp,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); hold on
semilogy(1:numel(loss_s),loss_s,'-','linewidth',2,'markersize',8,'markerfacecolor','w'); 
ylabel("Cross Entropy");grid on;xticklabels([]);
yticks(10.^(-7:1:0));
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
ylim([1e-3,70]);grid on;yticks(10.^(-2:1:1));
set(gca,'fontsize',15,'linewidth',1.5);

max(test_accuracy_hp)
max(test_accuracy_s)