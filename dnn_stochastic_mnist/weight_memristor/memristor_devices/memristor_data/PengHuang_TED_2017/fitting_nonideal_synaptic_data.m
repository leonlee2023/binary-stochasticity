clc
clear
close all

data=importdata('rram_synaptic_reset.txt');
Rdep=data.data(2:end,2);
Ndep=(1:numel(Rdep))-1;

line1=plot(Ndep,1./Rdep,'-s','color','b','linewidth',2,'markersize',10,'markerfacecolor','w');hold on;


Gmax=2.5e-4;
Gmin=2e-5;

% weight update during potentiation
N_d=100;
alpha_d=15;

if alpha_d == 0
    alpha_d = 1e-6;
end

for nn=1:5
 
    G=zeros(1,N_d+1);
    G(1)=Gmax;

    
    gamma0=0.3;
    for n=1:N_d
        dG = -((Gmax-Gmin)/(1-exp(-alpha_d))-(Gmax-G(n)))*(1-exp(-alpha_d/N_d));
        G(n+1) = G(n)+dG*(1+gamma0*randn());
    end
    
    G(G>Gmax)=Gmax;
    G(G<Gmin)=Gmin;
    plot(0:N_d,G,'-','color',[0.8,0.8,0.8],'linewidth',2,'markerfacecolor','w');hold on;
end

gamma0=0;

G=zeros(1,N_d+1);
G(1)=Gmax;

for n=1:N_d
    dG = -((Gmax-Gmin)/(1-exp(-alpha_d))-(Gmax-G(n)))*(1-exp(-alpha_d/N_d));
    G(n+1) = G(n)+dG*(1+gamma0*randn());
end

plot(0:N_d,G,'-','color','r','linewidth',2,'markerfacecolor','w');hold on;

xlabel('Pulse Number');
ylabel('Conductance [S]');
set(gca,'fontsize',20,'linewidth',3);

ylim([5e-6,3e-4])
xlim([0,100])
af=gcf;
af.Position=[488 501 560 260];
box off
legend(line1,'OxRRAM')