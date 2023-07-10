clc
clear
close all

type={'HP','S'}; 

TYPE={'ReLU','Truncated-ReLU','Truncated-ReLU2','logistic-ReLU','ReLU-logistic'};

test_accuracy_all=zeros(5,8);

for tt=1:numel(TYPE)
    act_type=TYPE{tt};
    folder=sprintf('results_%s_ep1000/',act_type);

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
                test_accuracy_all(tt,n)=max(test_accuracy);
            end
        end
    end

end

fig=figure;
fig.Position=[426 557 1231 420];
cat = categorical(TYPE);
cat = reordercats(cat,TYPE);
bar(cat,test_accuracy_all*100)
ylim([95,100]);grid on;
ylabel("Test Accuracy [%]");legend(leg_str)
set(gca,'fontsize',15,'linewidth',1.5);






