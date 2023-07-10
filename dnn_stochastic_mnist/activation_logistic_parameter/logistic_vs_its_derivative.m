clc
clear
close all

fig=figure;
x=linspace(-6,6,1000);

y1=1./(1+exp(-4*x));

y2=4*y1.*(1-y1);

plot(x,y1,'-','color','r','linewidth',4);hold on;
plot(x,y2,'-','color','b','linewidth',4);
set(gca,'color','none')
set(gca,'fontsize',25,'linewidth',3);
ylabel("z,dz/dy");xlabel('y');

legend({'\sigma(4*y)','dz/dy'},'fontsize',25,'location','northwest')