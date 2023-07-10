clc
clear
close all

type={'HP','S'}; % 'sampling', 'analog', 'binary':activation type for forwad pass output

fig1=figure;
syms={'s','o','^','<','>','d','s','o','^','<','>','d'};
leg_str={};
test_accuracy_all=[];
n=0;
for ff=1:numel(type)
    for ee=1:numel(type)
        for dd=1:numel(type)
            n=n+1;

            filename=sprintf('*F%s_E%s_D%s_*.mat',type{ff},type{ee},type{dd});
            file = dir(filename);
            load(file.name)
            test_accuracy=dnn.test_accuracy;
            fig1;
            if n<=4
                semilogy(1:numel(test_accuracy),(1-test_accuracy)*100,'-','marker','none','linewidth',2,'markersize',5,'markerfacecolor','w'); hold on
            else
                semilogy(1:numel(test_accuracy),(1-test_accuracy)*100,'-.','marker','none','linewidth',2,'markersize',5,'markerfacecolor','w'); hold on
            end
            leg_str{n}=sprintf('F:%s, E:%s, D:%s',type{ff},type{ee},type{dd});
            test_accuracy_all=[test_accuracy_all,max(test_accuracy)];
        end
    end
end

fig1;
xlabel('Epoch');ylabel("Test Error [%]");
ylim([1,3]);grid on;
set(gca,'fontsize',15,'linewidth',1.5); 
legend(leg_str,'location','northeast');

fig2=figure;
fig2.Position=[957 447 903 419];
cat = categorical(leg_str);
cat = reordercats(cat,leg_str);
bar(cat,test_accuracy_all*100)
ylim([95,100]);grid on;
ylabel("Test Accuracy [%]");
set(gca,'fontsize',15,'linewidth',1.5);

savefig(fig1,'test_error_epoch.fig') 
savefig(fig2,'test_accuracy_comparison.fig') 

%%
fig3=figure;
leg_str={};
train_accuracy_all=[];
n=0;
for ff=1:numel(type)
    for ee=1:numel(type)
        for dd=1:numel(type)
            n=n+1;

            filename=sprintf('*F%s_E%s_D%s_*.mat',type{ff},type{ee},type{dd});
            file = dir(filename);
            load(file.name)
            train_accuracy=dnn.train_accuracy;
            if n<=4
                semilogy(1:numel(train_accuracy),(1-train_accuracy)*100,'-','marker','none','linewidth',2,'markersize',5,'markerfacecolor','w'); hold on
            else
                semilogy(1:numel(train_accuracy),(1-train_accuracy)*100,'-.','marker','none','linewidth',2,'markersize',5,'markerfacecolor','w'); hold on
            end
            leg_str{n}=sprintf('F:%s, E:%s, D:%s',type{ff},type{ee},type{dd});
            train_accuracy_all=[train_accuracy_all,max(train_accuracy)];
        end
    end
end
xlabel('Epoch');ylabel("Test Error [%]");
ylim([1e-3,10]);grid on;
set(gca,'fontsize',15,'linewidth',1.5); 
legend(leg_str,'location','northeast');
%%
fig3=figure;
leg_str={};
n=0;
for ff=1:numel(type)
    for ee=1:numel(type)
        for dd=1:numel(type)
            n=n+1;

            filename=sprintf('*F%s_E%s_D%s_*.mat',type{ff},type{ee},type{dd});
            file = dir(filename);
            load(file.name)
            loss=dnn.Loss_epoch;
            if n<=4
                semilogy(1:numel(loss),loss,'-','marker','none','linewidth',2,'markersize',5,'markerfacecolor','w'); hold on
            else
                plot(1:numel(loss),loss,'-.','marker','none','linewidth',2,'markersize',5,'markerfacecolor','w'); hold on
            end
            leg_str{n}=sprintf('F:%s, E:%s, D:%s',type{ff},type{ee},type{dd});
        end
    end
end
xlabel('Epoch');ylabel("Cross Entropy Loss");
ylim([1e-5,10]);
grid on;
set(gca,'fontsize',15,'linewidth',1.5); 
legend(leg_str,'location','northeast');
%%

test_accuracy_all=[];
n=0;
for ff=1:numel(type)
    for ee=1:numel(type)
        for dd=1:numel(type)
            n=n+1;
            filename=sprintf('*F%s_E%s_D%s_*.mat',type{ff},type{ee},type{dd});
            file = dir(filename);
            load(file.name)

            figure;
            leg_str2=cell(1,dnn.n_layers);
            for ll=1:dnn.n_layers
                weight=reshape(dnn.nn(ll).weight,1,numel(dnn.nn(ll).weight));
                [f,xi]=ksdensity(weight);
                plot(xi,f,'linewidth',3);hold on;
                leg_str2{ll}=['w_',num2str(ll)];
            end
            set(gca,'fontsize',20,'linewidth',2);             
            ylabel("Probability Density");
            xlabel('Weight');
            title(leg_str{n});
            legend(leg_str2);
            xlim([-1,1])
        end
    end
end
