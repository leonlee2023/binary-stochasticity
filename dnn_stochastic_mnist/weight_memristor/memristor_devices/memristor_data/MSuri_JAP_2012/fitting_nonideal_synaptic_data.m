clc
clear
close all

data=importdata('pcm_potentiation.txt');
Gpot=data.data(:,2);
Npot=(1:numel(Gpot))-1;

line1=plot(Npot,Gpot,'-s','color','b','linewidth',2,'markersize',10,'markerfacecolor','w');hold on;

data=importdata('pcm_depression.txt');
Gdep=data.data(:,2);
Ndep=(1:numel(Gdep))-1;

%plot(Ndep+Npot(end),Gdep,'-s','color','b','linewidth',2,'markersize',10,'markerfacecolor','w');



Gmax=2.2e-3;
Gmin=7e-6;

% weight update during potentiation
N_p=30;
alpha_p=6;
N_d=0;
alpha_d=50;

if alpha_p == 0
    alpha_p = 1e-6;
end

if alpha_d == 0
    alpha_d = 1e-6;
end

for nn=1:5
    gamma0=0.3;
    
    G=zeros(1,N_p+N_d+1);
    G(1)=Gmin;
    for n=1:N_p
        dG = ((Gmax-Gmin)/(1-exp(-alpha_p))-(G(n)-Gmin))*(1-exp(-alpha_p/N_p));
        G(n+1) = G(n)+dG*(1+gamma0*randn());
    end
    
    gamma0=0.1;
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

ylim([5e-6,2.5e-3])
af=gcf;
af.Position=[488 501 560 260];
box off
legend(line1,'PCM')