function figure_msb=synpatic_behaivor(mem)

figure_msb=figure;

if mem.alpha_p == 0
    mem.alpha_p = 1e-6;
end

if mem.alpha_d == 0
    mem.alpha_d = 1e-6;
end

for nn=1:20 
    G=zeros(1,mem.Np+mem.Nd+1);
    if mem.Np ~= 0
        G(1)=mem.Gmin;
    else
        G(1)=mem.Gmax;
    end
    for n=1:mem.Np
        dG = ((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_p))-(G(n)-mem.Gmin))*(1-exp(-mem.alpha_p/mem.Np));
        G(n+1) = G(n)+dG*(1+mem.gamma_p*randn());
    end
    
    for n=mem.Np+1:mem.Np+mem.Nd
        dG = -((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_d))-(mem.Gmax-G(n)))*(1-exp(-mem.alpha_d/mem.Nd));
        G(n+1) = G(n)+dG*(1+mem.gamma_d*randn());
    end
    
    G(G>mem.Gmax)=mem.Gmax;
    G(G<mem.Gmin)=mem.Gmin;
    plot(0:mem.Np+mem.Nd,G,'-','color',[0.8,0.8,0.8],'linewidth',2,'markerfacecolor','w');hold on;
end

gamma=0;

G=zeros(1,mem.Np+mem.Nd+1);
if mem.Np ~= 0
    G(1)=mem.Gmin;
else
    G(1)=mem.Gmax;
end
for n=1:mem.Np
    dG = ((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_p))-(G(n)-mem.Gmin))*(1-exp(-mem.alpha_p/mem.Np));
    G(n+1) = G(n)+dG*(1+gamma*randn());
end

for n=mem.Np+1:mem.Np+mem.Nd
    dG = -((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_d))-(mem.Gmax-G(n)))*(1-exp(-mem.alpha_d/mem.Nd));
    G(n+1) = G(n)+dG*(1+gamma*randn());
end

plot(0:mem.Np+mem.Nd,G,'-','color','r','linewidth',2,'markerfacecolor','w');hold on;

xlabel('Pulse Number');
ylabel('Conductance, G [S]');
set(gca,'fontsize',15,'linewidth',2);

ylim([mem.Gmin,mem.Gmax])

drawnow