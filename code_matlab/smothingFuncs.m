%SMOTHING

function funcs = smothingFuncs()
 funcs  = {@none,@smothing_sgolay,@smothing_mw_average};
end

function [NewXtrain,NewXval] = smothing_mw_average(Xtrain,Xval)
NewXtrain = smoothdata(Xtrain,2,'movmean',default_windows());
if nargin>1
    NewXval = smoothdata(Xval,2,'movmean',default_windows());
end
end

function [NewXtrain,NewXval] = smothing_sgolay(Xtrain,Xval)
NewXtrain = smoothdata(Xtrain,2,'sgolay',default_windows());
if nargin>1
    NewXval = smoothdata(Xval,2,'sgolay',default_windows());
end
end
