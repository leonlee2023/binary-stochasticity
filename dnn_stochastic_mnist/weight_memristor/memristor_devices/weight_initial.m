function w0= weight_initial(mem,Size)
    Gref=(mem.Gmin+mem.Gmax)/2;
    switch mem.wup_type
        case 'pot_dep'
            w0.G     = Gref*(1+0.1*randn(Size)); % positive weight part
            w0.Gref_col  = Gref*(1+0.1*randn([Size(1),1])); % reference weight part
            w0.Gref_row  = Gref*(1+0.1*randn([1,Size(2)])); % reference weight part
            w0.num_memristors=numel(w0.G)+numel(w0.Gref_col)+numel(w0.Gref_row);
        case 'pot_only'
            w0.Gpos     = mem.Gmin+0.1*mem.G0*randn(Size); % positive weight part
            w0.Gneg     = mem.Gmin+0.1*mem.G0*randn(Size); % negtive weight part
            w0.num_memristors=numel(w0.Gpos)+numel(w0.Gneg);
        case 'dep_only'
            w0.Gpos     = mem.Gmax*(0.95+0.05*randn(Size)); % positive weight part
            w0.Gneg     = mem.Gmax*(0.95+0.05*randn(Size)); % negtive weight part
            w0.num_memristors=numel(w0.Gpos)+numel(w0.Gneg);
        otherwise
            error('Weight update type not defined!')
    end
    w0.WupCount = zeros(Size); % Counter to count how many weight update operations are performed on each device
end