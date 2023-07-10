clc
clear
close all

TYPE={'logistic','ReLU','Truncated-ReLU','Truncated-ReLU2','logistic-ReLU','ReLU-logistic'};

for tt=1:numel(TYPE)
    type=TYPE{tt};
    fig=figure;
    a=1;
    if tt<=4
        xlims=[-2.5,2.5];
    else
        xlims=[-6,6];
    end
    y=linspace(xlims(1),xlims(2),1000);
    [z,dzdy]=activation(y,type,a);
    plot(y,z,'-','color','r','linewidth',3);hold on;
    plot(y,dzdy,'-','color','b','linewidth',3);
    set(gca,'color','none')
    set(gca,'fontsize',20,'linewidth',2);
    ylabel("z,dz/dy");xlabel('y');
    legend({'z','dz/dy'},'fontsize',20,'location','northwest')
    xlim(xlims)
end
