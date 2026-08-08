%scaling
function funcs = centeringFuncs()
funcs  = {@none,@centering_mean}; 
%funcs  = {@none}; 
end

function [NewXtrain,NewXval] = centering_mean(Xtrain,Xval)
[NewXtrain,mx] = mncn(Xtrain);
if nargin>1
    NewXval = Xval-mx;
end
end

