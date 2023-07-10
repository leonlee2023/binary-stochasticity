clc
clear
close all

type={'HP','S'}; % 'sampling', 'analog', 'binary':activation type for forwad pass output

Act_a=[1,2,4,8];
test_accuracy_all=zeros(4,8);

for da=1:numel(Act_a)
    act_a=Act_a(da);
    act_d=act_a;
    folder=sprintf('results_A%dD%d_ep1000/',act_a,act_d);
    
    leg_str={};
    n=0;
    for ff=1:numel(type)
        for ee=1:numel(type)
            for dd=1:numel(type)
                filename=sprintf([folder,'*F%s_E%s_D%s_*.mat'],type{ff},type{ee},type{dd});
                file = dir(filename);
                load([folder,file.name])
                test_accuracy=dnn.test_accuracy;
                n=n+1;
                leg_str{n}=sprintf('F:%s, E:%s, D:%s',type{ff},type{ee},type{dd});
                test_accuracy_all(da,n)=max(test_accuracy);
            end
        end
    end

end

fig=figure;
fig.Position=[426 557 1231 420];
labels={'z=\sigma(y)','z=\sigma(2*y)','z=\sigma(4*y)','z=\sigma(8*y)'};
cat = categorical(labels);
cat = reordercats(cat,labels);
bar(cat,test_accuracy_all*100)
ylim([95,100]);grid on;
ylabel("Test Accuracy [%]");legend(leg_str)
set(gca,'fontsize',15,'linewidth',1.5);


%%
clc
clear
close all

type={'HP','S'}; % 'sampling', 'analog', 'binary':activation type for forwad pass output

Act_d=[1,2,4,8];
test_accuracy_all=zeros(4,8);

for da=1:numel(Act_d)
    act_a=1;
    act_d=Act_d(da);
    folder=sprintf('results_A%dD%d/',act_a,act_d);
    
    leg_str={};
    n=0;
    for ff=1:numel(type)
        for ee=1:numel(type)
            for dd=1:numel(type)
                filename=sprintf([folder,'*F%s_E%s_D%s_*.mat'],type{ff},type{ee},type{dd});
                file = dir(filename);
                load([folder,file.name])
                test_accuracy=dnn.test_accuracy;
                n=n+1;
                leg_str{n}=sprintf('F:%s, E:%s, D:%s',type{ff},type{ee},type{dd});
                test_accuracy_all(da,n)=max(test_accuracy);
            end
        end
    end

end

fig=figure;
fig.Position=[426 557 1231 420];
labels={'dzdy=z(1-z)','dzdy=2*z(1-z)','dzdy=4*z(1-z)','dzdy=8*z(1-z)'};
cat = categorical(labels);
cat = reordercats(cat,labels);
bar(cat,test_accuracy_all*100)
ylim([90,100]);grid on;
ylabel("Test Accuracy [%]");legend(leg_str)
set(gca,'fontsize',15,'linewidth',1.5);





