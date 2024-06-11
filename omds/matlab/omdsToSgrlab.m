function c = omdsToSgrlab(file_name, varargin)
% omdsToSgrlab.m Convert  OMDS HDF5 file to sgrlab data structure.

root = '/';
g = h5info(file_name, root);

c = {};
for ii = 1:length(g.Groups)
    process_group(g.Groups(ii), root);
end
if strcmp(get_attribute(g, 'class'), 'Spectrum')
    process_spectrum(g, fullfile(root, g.Name))
end

    function ret = get_attribute(obj, name)
        idx = strcmp({obj.Attributes.Name},name);
        if any(idx)
            ret = obj.Attributes(idx).Value;
        else
            ret = [];
        end            
    end

    function process_group(group, root)
        for jj = 1:numel(group.Groups)
            gg = group.Groups;
            process_group(gg, fullfile(root, group.Name));
        end
        
        if strcmp(get_attribute(group, 'class'), 'Spectrum')
            process_spectrum(group, fullfile(root, group.Name))
        end
    end

    function process_spectrum(spectrum, root)
        order = get_attribute(spectrum, 'order');
        switch order
            case 3
                process_order_3(spectrum, root)
            otherwise
                error('SGRLab:NotImplementedError',...
                    ['Order ', num2str(order), ' not yet implemented'])
        end
    end

    function process_order_3(spectrum, root)
        order = 3;
        x = cell([1, order]);
        x_units = cell([1, order]);
        p = cell([1, order+1]);
        
        for jj = 1:numel(spectrum.Datasets)
            dset = spectrum.Datasets(jj);
            switch get_attribute(dset, 'class')
                case 'Response'
                    R = h5read(file_name, ...
                        fullfile(root, dset.Name));
                case 'Polarization'
                    m = regexp(dset.Name, 'pol(?<dim>\d+)','names');
                    kk = str2double(m.dim);
                    p{kk} = get_attribute(dset, 'label');
                case 'Axis'
                    m = regexp(dset.Name, 'x(?<dim>\d+)', 'names');
                    kk = str2double(m.dim);
                    x{kk} = h5read(file_name, ...
                        fullfile(root, dset.Name));
                    x_units{kk} = 
            end
        end
        d = construct2dPP;
        n_t2s = size(R,2);
        for jj = 1:n_t2s
            d(jj).R = squeeze(R(:,jj,:))';
            d(jj).w1 = x{1};
            d(jj).t2 = x{2}(jj);
            d(jj).w3 = x{3};
            d(jj).polarization = [p{:}];
        end
        c = [c; d];
    end
end

