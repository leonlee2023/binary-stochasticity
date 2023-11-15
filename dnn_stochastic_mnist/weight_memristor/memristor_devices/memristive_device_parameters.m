clc
clear
close all

% parameters fitting to data of ECRAM [JianshiTang, IEDM, 2018].
mem.name='JTang_IEDM_2018';
mem.device='ECRAM';
mem.publication='JTang, IEDM, 2018';
mem.wup_type = 'pot_dep';
mem.Gmax=3e-9;
mem.Gmin=1e-9;
mem.Vread=0.1;
mem.Np=55;
mem.Nd=55;
mem.alpha_p=0.5;
mem.alpha_d=0.5;
mem.gamma_p=0.3;
mem.gamma_d=0.3;

MEM(1)=mem;

% parameters fitting to data of PCM [M.Suri, JAP, 2012].
mem.name='MSuri_JAP_2012';
mem.device='PCM';
mem.publication='MSuri, JAP, 2012';
mem.wup_type = 'pot_only';
mem.Gmax=2.2e-3;
mem.Gmin=7e-6;
mem.Vread=0.1;
mem.Np=30;
mem.Nd=10;
mem.alpha_p=6;
mem.alpha_d=50;
mem.gamma_p=0.3;
mem.gamma_d=0.1;

MEM(2)=mem;

% parameters fitting to data of PCMO [Jang, EDL, 2015].
mem.name='Jang_EDL_2015';
mem.device='PCMO';
mem.publication='Jang, EDL, 2015';
mem.wup_type = 'pot_dep';
mem.Gmax=1.6e-7;
mem.Gmin=3.5e-8;
mem.Vread=0.1;
mem.Np=100;
mem.Nd=100;
mem.alpha_p=6;
mem.alpha_d=20;
mem.gamma_p=1;
mem.gamma_d=1;

MEM(3)=mem;

% parameters fitting to data of RRAM [Choi, Nature Materials, 2018].
mem.name='Choi_NM_2018_1';
mem.device='SiGe epiRAM-1';
mem.publication='Choi, NM, 2018(1)';
mem.wup_type = 'pot_dep';
mem.Gmax=4e-5;
mem.Gmin=1e-6;
mem.Vread=2;
mem.Np=500;
mem.Nd=400;
mem.alpha_p=8;
mem.alpha_d=15;
mem.gamma_p=2;
mem.gamma_d=2;

MEM(4)=mem;

% parameters fitting to data of RRAM [Choi, Nature Materials, 2018].
mem.name='Choi_NM_2018_2';
mem.device='SiGe epiRAM-2';
mem.publication='Choi, NM, 2018(2)';
mem.wup_type = 'pot_dep';
mem.Gmax=3e-5;
mem.Gmin=1e-7;
mem.Vread=2;
mem.Np=200;
mem.Nd=50;
mem.alpha_p=5;
mem.alpha_d=1;
mem.gamma_p=1;
mem.gamma_d=1;

MEM(5)=mem;

% parameters fitting to data of RRAM [Choi, Nature Materials, 2018].
mem.name='Choi_NM_2018_3';
mem.device='SiGe epiRAM-3';
mem.publication='Choi, NM, 2018(3)';
mem.wup_type = 'pot_dep';
mem.Gmax=1.25e-5;
mem.Gmin=1e-7;
mem.Vread=2;
mem.Np=100;
mem.Nd=50;
mem.alpha_p=1;
mem.alpha_d=1;
mem.gamma_p=1;
mem.gamma_d=1;

MEM(6)=mem;

% parameters fitting to data of RRAM [PengHuang, TED, 2017].
mem.name='PHuang_TED_2017';
mem.device='OxRAM';
mem.publication='PHuang, TED, 2017';
mem.wup_type = 'dep_only';
mem.Gmax=2.5e-4;
mem.Gmin=2e-5;
mem.Vread=0.1;
mem.Np=0;
mem.Nd=100;
mem.alpha_p=10;
mem.alpha_d=15;
mem.gamma_p=0.3;
mem.gamma_d=0.3;

MEM(7)=mem;

for mm=1:numel(MEM)
    synpatic_behaivor(MEM(mm));
    title(MEM(mm).publication);
    drawnow
end

save memristive_device_parameters.mat MEM