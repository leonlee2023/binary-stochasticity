function output=read_matrix(input_states,weight,mem,direction)

if ~exist('direction','var')
    direction = 'forward';
end

Vread=input_states*mem.Vread;
switch mem.wup_type
    case 'pot_dep'
        switch direction
            case 'forward'
                Iout = Vread*weight.G;
                Iref = Vread*weight.Gref_col; 
            case 'back'
                Iout = Vread*weight.G';
                Iref = Vread*weight.Gref_row';
        end
        
        output=(Iout-Iref)/mem.I0;

    case {'pot_only','dep_only'}
        switch direction
            case 'forward'
                Ipos = Vread*weight.Gpos;
                Ineg = Vread*weight.Gneg; 
            case 'back'
                Ipos = Vread*weight.Gpos';
                Ineg = Vread*weight.Gneg';
        end
        
        output=(Ipos-Ineg)/mem.I0;
    otherwise
        error('Weight update type not defined!')
end

