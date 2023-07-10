clc
clear
close all

data=importdata('synaptic_weight_updates.txt');
Gwup=data.data(:,2);
Npulse=data.data(:,1);

line1=plot(Npulse,Gwup,'-s','color','b','linewidth',2,'markersize',10,'markerfacecolor','w');hold on;

Gmax=1.6e-7;
Gmin=3.5e-8;

% weight update during potentiation
N_p=100;
alpha_p=6;
N_d=100;
alpha_d=20;

if alpha_p == 0
    alpha_p = 1e-6;
end

if alpha_d == 0
    alpha_d = 1e-6;
end

for nn=1:5
    gamma0=1;
    
    G=zeros(1,N_p+N_d+1);
    G(1)=Gmin;
    for n=1:N_p
        dG = ((Gmax-Gmin)/(1-exp(-alpha_p))-(G(n)-Gmin))*(1-exp(-alpha_p/N_p));
        G(n+1) = G(n)+dG*(1+gamma0*randn());
    end
    
    gamma0=1;
    for n=N_p+1:N_p+N_d
        dG = -((Gmax-Gmin)/(1-exp(-alpha_d))-(Gmax-G(n)))*(1-exp(-alpha_d/N_d));
        G(n+1) = G(n)+dG*(1+gamma0*randn());
    end
    
    G(G>Gmax)=Gmax;
    G(G<Gmin)=Gmin;
    plot(0:N_p+N_d,G,'-','color',[0.8,0.8,0.8],'linewidth',2,'markerfacecolor','w');hold on;
end

gamma0=0;

G=zeros(1,N_p+N_d+1);
G(1)=Gmin;
for n=1:N_p
    dG = ((Gmax-Gmin)/(1-exp(-alpha_p))-(G(n)-Gmin))*(1-exp(-alpha_p/N_p));
    G(n+1) = G(n)+dG*(1+gamma0*randn());
end

for n=N_p+1:N_p+N_d
    dG = -((Gmax-Gmin)/(1-exp(-alpha_d))-(Gmax-G(n)))*(1-exp(-alpha_d/N_d));
    G(n+1) = G(n)+dG*(1+gamma0*randn());
end

plot(0:N_p+N_d,G,'-','color','r','linewidth',2,'markerfacecolor','w');hold on;

xlabel('Pulse Number');
ylabel('Conductance [S]');
set(gca,'fontsize',20,'linewidth',3);
legend(line1,'PCMO')

ylim([2e-8,1.8e-7])
af=gcf;
af.Position=[488 501 560 260];
box off