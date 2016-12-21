clear all;
load('EMIHW6.mat');   %Loading file with input
X = EMIHW6;

%Setting Network structure parameters

L1 = 2,L2=4,L3=1,H_layers = 1;;     % L1,L2,L3 : size of input,hidden and output layer without bias

total_size= L2*(L1 +1 ) + L3* ( L2 + 1);     % Defining parameter vector size
%total_size = L2*(L1+1) + L3*(L2+1) + (H_layers-1)*L2*(L2+1);
theta = rand(total_size,1);             % Initialising random weights
Y = X(:,3);                             % output
Parameter1 = reshape(theta((1:L2*(L1+1)),:),[L2 L1+1]);                     % Generating weight matrices
Parameter2 = reshape(theta(L2*(L1+1)+1:L2*(L1+1)+L3*(L2+1),:),[L3 L2+1]);
for iteration =1:10000
    output_mat = zeros(size(Y,1),1);         %Matrix for storing output
  Accum_par1 =zeros(L2,L1+1);              % Storing accumulated values after each batch is passed
    Accum_par2 = zeros(L3,L2+1);
    
    for i =1:8              % 8 samples in training batch
        Lone = X(i,(1:2))';       % Lone, Ltwo, Lthree layers without bias
        Ltwo = zeros(L2,1) ;
        Lthree = zeros(L3,L3);
        
        % Forward Propogation ,A1,A2,A3 layers with bias
        
        [A1,A2,A3,output] = forwardprop(Lone,Parameter1,Parameter2);  %Lone and Parameter1 and Parameter2 passed as arguement for
        
        output_mat(i,:) = output;      % output stored
        Ltwo = A2(2:size(A2,1),:);
        Lthree = A3;
        
        %Backward Propogation
        
        e3 = output - Y(i) ;
        [e2] = backward_prop(e3,Parameter2,A2);
        
        Accum_par1 = Accum_par1 + e2(2:size(e2, 1))*A1';
        Accum_par2 = Accum_par2 + e3*A2';
        
    end
    
    % calculating D1 & D2 which are partial dervatives w.r.t cost function
    
    D1 = Accum_par1*(1/size(X,1));   
    D2 = Accum_par2*(1/size(X,1));
    
   
       step_size = 2;                       %setting step_size    
       
       %Updating parameters
    Parameter1 = Parameter1 - step_size.*D1;
    Parameter2 = Parameter2 - step_size.*D2;
    
    %Cost Function (cross-entropy)
    cost(iteration,:) = (-1/size(X,1)*(log(output_mat)'*Y + log(1-output_mat)'*(1-Y)));
end








