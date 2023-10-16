from quantize_filters import *


#File for quantized linear layer

class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                    num_bits_w=8, num_bits_a=8,
                    quant_w=False, quant_a=False, quant_a_type='per_layer'):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.bitsw = num_bits_w
        self.bitsa = num_bits_a
        self.quant_w = quant_w
        self.quant_a = quant_a
        self.quant_a_type = quant_a_type
        assert self.quant_a_type in ['per_layer','per_channel']

        self.set_agregation(1)

        self.alpha = nn.Parameter(torch.tensor(3.0),requires_grad=False)
        self.register_buffer('alpha_init', torch.zeros(1))
        self.bias_corr = False

        self.quant_basis=False
        self.bitsb=8
        self.per_layer = False
        self.quantile = 0.9999

        self.bit_alloc_w = False
        self.map_bitsw = torch.ones(self.out_features)*self.bitsw
        self.bit_alloc_a = False
        self.map_bitsa = torch.ones(self.in_features)*self.bitsa
        self.d = {}
        

    def set_alpha(self,alpha):
        self.alpha=alpha

    def set_agregation(self, agreg):
        assert self.weight.numel() % agreg == 0
        self.agreg = agreg

    def set_bits(self,bits_w,bits_a):
        self.bitsw = bits_w
        self.bitsa = bits_a
    
    def broadcast_quantiles(self,n):
        return(self.d[int(n)])
    
    def compute_bit_alloc_policy_w(self):
        if self.bit_alloc_w:
            B = self.out_features*self.bitsw
            Bw = self.out_features*2**self.bitsw #budget for weights
            w = self.weight
            w = w.flatten(1).abs()
            alphas = (torch.max(w,dim=1)[0]-torch.min(w,dim=1)[0])**(2/3)
            self.map_bitsw = torch.round(torch.log2(Bw*alphas/(alphas.sum())))
            self.map_bitsw[self.map_bitsw<=0] = 0
            it = 0
            coef = 0
            while B > self.map_bitsw.sum() and it<50: #dichotomy to optimize number of bins in bit allocation scheme
                beta = torch.log2(Bw*alphas/(alphas.sum())) + coef + (1/2)**it
                testmap = torch.round(beta)
                testmap[testmap<=0] = 0
                if B >= testmap.sum():
                    coef+=(1/2)**it
                    self.map_bitsw = testmap
                it+=1
            it = 0
            coef = 0
            while B < self.map_bitsw.sum() and it<50: #in case basic ACIQ grants too many bits in total to the layer
                beta = torch.log2(Bw*alphas/(alphas.sum())) - coef - (1/2)**it
                testmap = torch.round(beta)
                testmap[testmap<=0] = 0
                if B <= testmap.sum():
                    coef+=(1/2)**it
                    self.map_bitsw = testmap
                it+=1
            assert(self.map_bitsw.sum()<=B) #check bit allocation
            assert(self.map_bitsw.sum() == torch.abs(self.map_bitsw).sum()) #check that bit alloc remains positive
        else:
            self.map_bitsw = torch.ones(self.out_features)*self.bitsw
    
    def compute_bit_alloc_policy_a(self,alphas):
        if self.bit_alloc_a:
            B = torch.numel(alphas)*self.bitsa
            Ba = torch.numel(alphas)*2**self.bitsa #budget for activations
            alphas = alphas**(2/3)
            #print(torch.log2(Ba*alphas/(alphas.sum())).flatten())
            self.map_bitsa = torch.round(torch.log2(Ba*alphas/(alphas.sum())))
            self.map_bitsa[self.map_bitsa<=0] = 0
            it = 0
            coef = 0
            while B > self.map_bitsa.sum() and it<50: #dichotomy to optimize number of bins in bit allocation scheme
                beta = torch.log2(Ba*alphas/(alphas.sum())) + coef + (1/2)**it
                testmap = torch.round(beta)
                testmap[testmap<=0] = 0
                if B >= testmap.sum():
                    coef+=(1/2)**it
                    self.map_bitsa = testmap
                it+=1
            it = 0
            coef = 0
            while B < self.map_bitsa.sum() and it<50: #in case basic ACIQ grants too many bits in total to the layer
                beta = torch.log2(Ba*alphas/(alphas.sum())) - coef - (1/2)**it
                testmap = torch.round(beta)
                testmap[testmap<=0] = 0
                if B <= testmap.sum():
                    coef+=(1/2)**it
                    self.map_bitsa = testmap
                it+=1
            assert(self.map_bitsa.sum()<=B) #check bit allocation
            assert(self.map_bitsa.sum() == torch.abs(self.map_bitsa).sum()) #check that bit alloc remains positive
        else:
            self.map_bitsa = torch.ones(self.in_features)*self.bitsa
            self.map_bitsa = self.map_bitsa.unsqueeze(0)

    def calibrate_base_per_channel(self, steps, step_lengths,init_basis, learn=False): #Calibration function for basis in case of per channel quantization of weights

        denom = torch.max(torch.abs(self.weight.detach()))
        filters_FP = self.weight / denom

        B = (torch.eye(self.agreg, dtype=torch.float)*torch.ones((filters_FP.shape[0],self.agreg,self.agreg))).cuda()
        B = 10**(-4) * B / (2**(self.bitsw-1))

        
        score1 = loss_torch(self.map_bitsw,filters_FP, B, self.bias_corr and self.bitsw<4).detach()

        if init_basis:
            self.compute_bit_alloc_policy_w()
        for j in range(len(steps)):
            T = steps[j]
            for k in range(step_lengths[j]):
                testbasis = transform_ball_torch(B,T/2,False)
                if self.quant_basis:
                    testbasis = quantize_basis_channel(testbasis,self.bitsb)
                score2 = loss_torch(self.map_bitsw,filters_FP, testbasis, self.bias_corr and self.bitsw<4).detach() #and self.bitsw<4
                score2[score2.isnan()]=float('inf') #happens sometimes when testbases not linearly independent (unlucky), ignore tested basis
                
                score = torch.le(score1,score2).unsqueeze(dim=0) #x element is True if score1[x]<= score2[x]
                score = torch.cat((score,torch.logical_not(score)),dim=0).int()
                
                score2[score2==float('inf')] = 0 #avoid nans due to inf*0
                score1 = (torch.cat((score1.unsqueeze(dim=0),score2.unsqueeze(dim=0)),dim=0)*score).sum(dim=0)
                score = score.unsqueeze(dim=2).unsqueeze(dim=2)
                scorebase = score*torch.ones((self.agreg,self.agreg)).cuda()
                bases = torch.cat((B.unsqueeze(dim=0),testbasis.unsqueeze(dim=0)),dim=0)
                bases = bases*scorebase
                B = torch.sum(bases,dim=0)
        
        if init_basis:
            print("MCE = " + str(score1.mean().item()))
            self.base = nn.Parameter(B,requires_grad=False) # Turn requires_grad to False not to learn the base
        else:
            previous_score = loss_torch(self.map_bitsw,filters_FP, self.base, self.bias_corr and self.bitsw<4).detach()
            previous_B = self.base
            score = torch.le(previous_score,score1).unsqueeze(dim=0) #x element is True if score1[x]<= score2[x]
            score = torch.cat((score,torch.logical_not(score)),dim=0).int()
            new_score = (torch.cat((previous_score.unsqueeze(dim=0),score1.unsqueeze(dim=0)),dim=0)*score).sum(dim=0)
            print("MCE = " + str(new_score.mean().item()))
            score = score.unsqueeze(dim=2).unsqueeze(dim=2)
            scorebase = score*torch.ones((self.agreg,self.agreg)).cuda()
            bases = torch.cat((previous_B.unsqueeze(dim=0),B.unsqueeze(dim=0)),dim=0)
            bases = bases*scorebase
            self.base = nn.Parameter(torch.sum(bases,dim=0),requires_grad=False)

        if torch.max(self.base).isnan():
            print('nan detected')


    def calibrate_base_per_layer(self,steps, step_lengths,init_basis, learn=False):

        denom = torch.max(torch.abs(self.weight.detach()))
        filters_FP = self.weight / denom
        filters_FP = filters_FP.flatten()
        filters_FP = filters_FP.reshape(-1, self.agreg)
        #print(filters_FP.shape)

        B = torch.eye(self.agreg, dtype=torch.float).cuda()
        B = B / ((2**(self.bitsw-1))-1)

        minloss = loss_torch_linear(self.bitsw,filters_FP, B, False)


        for j in range(len(steps)):
            T = steps[j]
            #print("T="+str(T))
            for k in range(step_lengths[j]):
                testbasis = transform_ball_torch(B,T/2)
                if self.quant_basis:
                    testbasis = quantize_basis_layer(testbasis,self.bitsb)
                filters_FP_test = filters_FP
                
                l = loss_torch_linear(self.bitsw,filters_FP, testbasis, False)
                if l<minloss:
                    minloss=l
                    B = testbasis
        print("MCE = " + str(minloss.item()))
        
        if init_basis:
            self.base = nn.Parameter(B,requires_grad=False)
        else:
            if minloss < loss_torch_linear(self.bitsw,filters_FP, self.base, self.bias_corr and self.bitsw<4).detach():
                self.base = nn.Parameter(B,requires_grad = False)
        

    def bias_correction(self,weight):
        FP_w = self.weight.detach()
        w = weight.detach()
        ones = torch.ones_like(w)
        mu = (FP_w-w).mean(dim=1,keepdim=True)*ones
        eps = 10**(-6)
        d = torch.linalg.norm((w - w.mean(dim=1,keepdim=True)*ones),dim=1,keepdim=True)+eps 
        inv = 1/d
        inv[inv.isnan()] = 0
        eta = torch.linalg.norm((FP_w - FP_w.mean(dim=1,keepdim=True)*ones),dim=1,keepdim=True)*inv

        w = eta*(w+mu)
        return(w)

    def forward(self, inp):
        if self.quant_w: 
            ### Clamp weights in range [0,1]

            denom = torch.max(torch.abs(self.weight.detach()))
            weight = self.weight / denom

            ### Quant weights by CVP
            weight = weight.reshape(weight.shape[0],-1,self.agreg)
            basis = self.base
            # Ortho process with gram_schmidt
            if self.per_layer:
                ortho_B = gram_schmidt_torch(basis)
                # Find closest vector with babai
                coord = babai_torch(self.bitsw, basis, ortho_B, weight)
            else:
                ortho_B = gram_schmidt_torch_channel(basis)
                # Find closest vector with babai
                coord = babai_torch_channel(self.map_bitsw, basis, ortho_B, weight)
                
            if not torch.all(torch.le(torch.floor(-2**(self.map_bitsw-1)).cuda()+1,coord.flatten(1).min(dim=1)[0])).item():
                print(torch.floor(-2**(self.map_bitsw-1)).cuda()+1)
                print(coord.flatten(1).min(dim=1)[0])

            #check quantization (lower bound)
            assert(torch.all(torch.le(torch.floor(-2**(self.map_bitsw-1)).cuda()+1,coord.flatten(1).min(dim=1)[0])).item()), "lower bound exceeded"
            #check quantization (upper bound)
            assert(torch.all(torch.le(coord.flatten(1).max(dim=1)[0],torch.floor(2**(self.map_bitsw-1)).cuda())).item()), "upper bound exceeded"

            weight = coordinates_to_vect_torch(basis,coord)
            weight = weight.reshape_as(self.weight)

            weight = weight*denom
        
        else:
            ### FP weights
            weight = self.weight

        
        if self.quant_a: #activations quantization
            input_val = inp
            if self.alpha_init == 0:
                
                if self.quant_a_type=='per_layer':

                    alpha = input_val.max() #very naive
                    self.map_bitsa = (torch.ones(self.in_features)*self.bitsa).unsqueeze(0)

                elif self.quant_a_type=='per_channel':

                    alpha = input_val.transpose(0,1) #get activations per channel
                    quantile = self.quantile #temporary quantile
                    alpha = torch.quantile(alpha,quantile,dim=1,keepdim=True) #per channel quantile
                    alpha = alpha.transpose(0,1) #reshape alpha
                    self.compute_bit_alloc_policy_a(alpha)
                    quantile = self.map_bitsa.flatten().clone().cpu().apply_(self.broadcast_quantiles).cuda() #quantile per channel
                    alpha = input_val.transpose(0,1) #get activations per channel
                    alpha = torch.diagonal(torch.quantile(alpha,quantile,dim=1)).unsqueeze(1) #per channel quantile
                    alpha = alpha.transpose(0,1) #reshape alpha

                alpha[alpha==0] = 1 #arbitrary value when quantile is 0
                self.alpha = torch.nn.Parameter(alpha,requires_grad=False)
                self.alpha_init.fill_(1)

            input_val = torch.clamp(input_val/self.alpha,0,1)  
            bins = (2**self.map_bitsa-1).cuda()
            input_val = 1/bins * (torch.round(input_val*bins))
            input_val = input_val*(self.alpha)
            
        else :
            '''
            Activation Full precision
            '''
            input_val = inp

        
        output = F.linear(input_val, weight, self.bias)
        return output