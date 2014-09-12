function T = utrivec2mat(v)
% 
% utrivec2mat([1,2,3,4,5,6])
% 
% ans =
% 
%      1     2     3
%      2     4     5
%      3     5     6

n = (-1 + sqrt(1 + 8*length(v)))/2;

T = zeros(n,n);       
vptr = 1;           
for i = 1 : n;
   j = n-i+1;     
   T(i, end-j+1:end) = v(vptr:vptr+j-1);
   vptr = vptr + j;
end

for i = 1 : n
    for j = i : n
        T(j,i) = T(i,j);
    end
end
