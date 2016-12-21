  X1 = randi([-1,1]*10^1,[100000,2])/10^1;
  output_mat = X1;
for i =1:size(X1,1)
 Lone = X1(i,(1:2))';       % Lone, Ltwo, Lthree layers without bias
 Ltwo = zeros(L2,1) ;   
 Lthree = zeros(L3,L3); 
 
[A1,A2,A3,output] = forwardprop(Lone,Parameter1,Parameter2);  %Lone and Parameter1 and Parameter2 passed as arguement
output_mat(i,3) = output;  
if output_mat(i,3) >= 0.5
    output_mat(i,4)=1;
else
    output_mat(i,4)=0;
end
end
gscatter(output_mat(:,1),output_mat(:,2),output_mat(:,4),'br','xo');




