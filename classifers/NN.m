function [Rec] = NN(tr_dat, tt_dat, trls, ttls)
dist = EuDist2(tt_dat',tr_dat');
[r c] = min(dist');
class = trls(c);
Rec = length(find(class-ttls==0))/length(ttls);
fprintf(['The classification result of 1NN is about ' num2str(Rec) '\n']);