function [A1,A2,A3,output] = forwardprop(Lone,Parameter1,Parameter2)
 A1 = [1;Lone];
 Z2 = Parameter1*A1;
 SZ2 =  sigmg(Z2);
 A2 = [1;SZ2];
 Z3 = Parameter2*A2;
 A3 =  sigmg(Z3);
 output = A3;
end
