clc
clear
close all

data=importdata('synaptic_weight_update.txt');
Gwup=data.data(:,2);
Npulse=data.data(:,1);

line1=plot(Npulse,Gwup*1e-6/2,'-s','color','b','linewidth',2,'markersize',10,'markerfacecolor','w');hold on;

data=importdata('synaptic_weights_update2.txt');
Gwup=data.data(:,2);
Npulse=data.data(:,1);

line2=plot(Npulse,Gwup*1e-6/2,'-o','color','b','linewidth',2,'markersize',10,'markerfacecolor','w');hold on;

data=importdata('synaptic_weights_update3.txt');
Gwup=data.data(:,2);
Npulse=data.data(:,1);

line3=plot(Npulse,Gwup/2,'-^','color','b','linewidth',2,'markersize',10,'markerfacecolor','w');hold on;


Gmax=8.0e-5/2;
Gmin=2e-6/2;

% weight update during potentiation
N_p=500;
alpha_p=8;
N_d=400;
alpha_d=15;

if alpha_p == 0
    alpha_p = 1e-6;
end

if alpha_d == 0
    alpha_d = 1e-6;
end

for nn=1:5
    gamma0=2;
    
    G=zeros(1,N_p+N_d+1);
    G(1)=Gmin;
    for n=1:N_p
        dG = ((Gmax-Gmin)/(1-exp(-alpha_p))-(G(n)-Gmin))*(1-exp(-alpha_p/N_p));
        G(n+1) = G(n)+dG*(1+gamma0*randn());
    end
    
    gamma0=2;
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

xlim([0,800])
ylim([1e-7,9e-5]/2)
af=gcf;
%af.Position=[488 342 800 420];
box off



% weight update during potentiation
Gmax=3e-5;
Gmin=1e-7;
N_p=200;
alpha_p=5;
N_d=50;
alpha_d=1;

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
    line4=plot(0:N_p+N_d,G,'-','color',[0.8,0.8,0.8],'linewidth',2,'markerfacecolor','w');hold on;
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

line4=plot(0:N_p+N_d,G,'-','color','r','linewidth',2,'markerfacecolor','w');hold on;


% 

% weight update during potentiation
Gmax=1.25e-5;
Gmin=1e-7;
N_p=100;
alpha_p=1;
N_d=50;
alpha_d=1;

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

legend([line1,line2,line3],{'SiGe epiRAM-1','SiGe epiRAM-2 ', 'SiGe epiRAM-3'},'fontsize',15,'linewidth',2)