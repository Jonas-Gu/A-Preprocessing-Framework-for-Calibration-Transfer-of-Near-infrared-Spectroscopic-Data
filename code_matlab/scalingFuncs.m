%scaling
function funcs = scalingFuncs()
funcs  = {@none,@scaling_auto};%@scaling_poisson,,@scaling_pareto,@scaling_sqrtmean,,
%funcs  = {@none};
end

function [NewXtrain,NewXval] = scaling_auto(Xtrain,Xval)
[NewXtrain,mx,stdx]  = auto(Xtrain);
if nargin>1
    NewXval = scale(Xval,mx,stdx);
end
if mean(stdx)<0.1
    NewXtrain = Xtrain;
    if nargin>1
       NewXval = Xval;
    end
end
end
