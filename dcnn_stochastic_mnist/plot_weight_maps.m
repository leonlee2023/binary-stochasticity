clc
clear
close all

load trained_dnn_conv_ep1000_eta0.10_HP_220829_0532.mat
% load trained_dnn_conv_ep1000_eta0.10_stochastic_220831_0837.mat

fig=figure;
for ll=1:dnn.n_conv+1
    if ll<=dnn.n_conv
        weight=reshape(dnn.conv(ll).weight,1,[]);
    else
        weight=reshape(dnn.full.weight,1,[]);
    end
    [f,xi]=ksdensity(weight);
    plot(xi,f,'linewidth',2);hold on;
    
end
set(gca,'fontsize',15,'linewidth',1.5);             
ylabel("Probability Density");
legend({'Conv 1','Conv 3','Full 4'})
xlim([-5,5]);
xticks([-5:1:5]);
xlabel('Weight');
fig.Position=[476 654 560 212];

%%

fig1=figure;
for im=1:dnn.conv(1).output_n_feature
    weight=dnn.conv(1).weight(:,:,:,im);
    subplot(1,8,im);
    imagesc(weight);
    axis off
end
fig1.Position=[476 803 554 63];

figure;
m=dnn.conv(2).input_n_feature;
n=dnn.conv(2).output_n_feature;
for i=1:dnn.conv(2).input_n_feature
    for j=1:dnn.conv(2).output_n_feature
        weight=dnn.conv(2).weight(:,:,i,j);
        subplot(m,n,(i-1)*n+j);
        imagesc(weight);
        axis off
    end
end



