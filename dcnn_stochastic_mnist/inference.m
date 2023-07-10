clc
clear
close all

load('../dataset/MNIST.mat')

% filename='trained_dnn_conv_ep1000_eta0.10_HP_220829_0532';
filename='trained_dnn_conv_ep1000_eta0.10_stochastic_220831_0837';
load([filename,'.mat'])

tic
test_acc_analog=test(images_ts,labels_ts,dnn,'HP');
fprintf('High precision forwarding: %.4f%%\n',test_acc_analog*100)
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
ylim([0.6,30]);grid on;
set(gca,'fontsize',15,'linewidth',1.5);
legend({'HP inference','Binary inference','Stochastic inference'},'location','northeast');


dnn.test_acc_analog=test_acc_analog;
dnn.test_acc_binary=test_acc_binary;
dnn.test_acc_sampling=test_acc_sampling;

save([filename,'.mat'],'dnn') 
savefig(fig_inf,[filename,'_inference.fig']) 

%%
close all
clear
clc

folder='';
filename='trained_dnn_conv_ep1000_eta0.10_HP_220829_0532';
load([folder,filename,'.mat'])
test_acc_analog_hp=dnn.test_acc_analog;
test_acc_binary_hp=dnn.test_acc_binary;
test_acc_sampling_hp=dnn.test_acc_sampling;
figure;
semilogy(1:numel(test_acc_sampling_hp),(1-test_acc_analog_hp)*ones(1,numel(test_acc_sampling_hp))*100,'-','linewidth',3,'markersize',12,'markerfacecolor','w');hold on 
plot(1:numel(test_acc_sampling_hp),(1-test_acc_binary_hp)*ones(1,numel(test_acc_sampling_hp))*100,'-','linewidth',3,'markersize',12,'markerfacecolor','w');hold on 
plot(1:numel(test_acc_sampling_hp),(1-test_acc_sampling_hp)*100,'-','linewidth',3,'markersize',10,'markerfacecolor','w'); 
xlabel('Inference repetition');ylabel("Inference Error [%]");
ylim([0.5,30]);grid on;
set(gca,'fontsize',15,'linewidth',1.5);
legend({'HP inference','Binary inference','Stochastic inference'},'location','east','LineWidth',0.1);


filename='trained_dnn_conv_ep1000_eta0.10_stochastic_220831_0837';
load([folder,filename,'.mat'])
figure;
test_acc_analog_s=dnn.test_acc_analog;
test_acc_binary_s=dnn.test_acc_binary;
test_acc_sampling_s=dnn.test_acc_sampling;
semilogy(1:numel(test_acc_sampling_s),(1-test_acc_analog_s)*ones(1,numel(test_acc_sampling_s))*100,'-','linewidth',3,'markersize',12,'markerfacecolor','w');hold on 
plot(1:numel(test_acc_sampling_s),(1-test_acc_binary_s)*ones(1,numel(test_acc_sampling_s))*100,'-','linewidth',3,'markersize',12,'markerfacecolor','w');hold on 
plot(1:numel(test_acc_sampling_s),(1-test_acc_sampling_s)*100,'-','linewidth',3,'markersize',10,'markerfacecolor','w'); 
xlabel('Inference repetition');ylabel("Inference Error [%]");
ylim([0.5,30]);grid on;
set(gca,'fontsize',15,'linewidth',1.5);
legend({'HP inference','Binary inference','Stochastic inference'},'location','east','LineWidth',0.1);

fig_comp=figure;
cat_names={'HP trained','Stochastically trained'};
cat = categorical(cat_names);
cat = reordercats(cat,cat_names);
acc = [test_acc_analog_hp,test_acc_binary_hp,test_acc_sampling_hp(1),test_acc_sampling_hp(25),test_acc_sampling_hp(end);
    test_acc_analog_s,test_acc_binary_s,test_acc_sampling_s(1),test_acc_sampling_s(25),test_acc_sampling_s(end)];
bar(cat,(1-acc)*100)
ylim([0.5,30]);
grid on;
ylabel("Inference Error [%]");
legend({'HP inference', ...
    'Binary inference',...
    'Stochastic inference (1)',...
    'Stochastic inference (25)',...
    'Stochastic inference (100)'},'LineWidth',0.1);
set(gca,'fontsize',15,'linewidth',1.5);
set(gca,'YScale','log')


