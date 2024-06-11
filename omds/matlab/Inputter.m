classdef (Abstract) Inputter

    methods
        function input(self, file_name)
            error('OMDS:Inputter:NotImplementedError',...
                'Subclass must implement an input function')
        end
    end
end