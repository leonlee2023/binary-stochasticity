function [w,DLDw]= weight_update(w0,DLDw,mem)

dw=-fix(DLDw/mem.DLDw_th);
DLDw=DLDw + dw * mem.DLDw_th;

if mem.alpha_d == 0
    mem.alpha_d = 1e-6;
end
if mem.alpha_p == 0
    mem.alpha_p = 1e-6;
end

switch mem.wup_type
    case 'pot_dep' 
         
        w=w0;
        dGp=((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_p))-(w.G(dw>0)-mem.Gmin))*(1-exp(-mem.alpha_p/mem.Np));
        dGp=dGp.*(1+mem.gamma_p*randn(size(dGp)));
        w.G(dw>0) = w.G(dw>0)+dGp;
        dGd=-((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_d))-(mem.Gmax-w.G(dw<0)))*(1-exp(-mem.alpha_d/mem.Nd));
        dGd=dGd.*(1+mem.gamma_d*randn(size(dGd)));
        w.G(dw<0) = w.G(dw<0)+dGd;
        
        w.G(w.G>mem.Gmax)=mem.Gmax;
        w.G(w.G<mem.Gmin)=mem.Gmin;
        
     case 'pot_only'
        
        w=w0;
        dGp=((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_p))-(w.Gpos(dw>0)-mem.Gmin))*(1-exp(-mem.alpha_p/mem.Np));
        dGp=dGp.*(1+mem.gamma_p*randn(size(dGp)));
        w.Gpos(dw>0) = w.Gpos(dw>0)+dGp;
        dGd=((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_p))-(w.Gneg(dw<0)-mem.Gmin))*(1-exp(-mem.alpha_p/mem.Np));
        dGd=dGd.*(1+mem.gamma_d*randn(size(dGd)));
        w.Gneg(dw<0) = w.Gneg(dw<0)+dGd;
        
        w.Gpos(w.Gpos>mem.Gmax)=mem.Gmax;
        w.Gneg(w.Gneg>mem.Gmax)=mem.Gmax;

        w.Gpos(w.Gpos<mem.Gmin)=mem.Gmin;
        w.Gneg(w.Gneg<mem.Gmin)=mem.Gmin;

     case 'dep_only' 
         
        w=w0;
        dGp=-((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_d))-(mem.Gmax-w.Gneg(dw>0)))*(1-exp(-mem.alpha_d/mem.Nd));
        dGp=dGp.*(1+mem.gamma_d*randn(size(dGp)));
        w.Gneg(dw>0) = w.Gneg(dw>0)+dGp;
        dGd=-((mem.Gmax-mem.Gmin)/(1-exp(-mem.alpha_d))-(mem.Gmax-w.Gpos(dw<0)))*(1-exp(-mem.alpha_d/mem.Nd));
        dGd=dGd.*(1+mem.gamma_d*randn(size(dGd)));
        w.Gpos(dw<0) = w.Gpos(dw<0)+dGd;
        
        w.Gpos(w.Gpos>mem.Gmax)=mem.Gmax;
        w.Gneg(w.Gneg>mem.Gmax)=mem.Gmax;
        
        w.Gpos(w.Gpos<mem.Gmin)=mem.Gmin;
        w.Gneg(w.Gneg<mem.Gmin)=mem.Gmin;
end

w.WupCount = w.WupCount + abs(dw);

