
from quantize_filters import *


#File for quantized convolution layer

class QConv2d(nn.Conv2d): #quantized convolution layer
    """
        QConv2d: This is the quantization module of Conv2d    
    """

    def __init__(self, in_channels, out_channels, 
                    kernel_size, stride=1, padding=0, 
                    dilation=1, groups=1, bias=True,
                    num_bits_w=8, num_bits_a=8,
                    quant_w=False, quant_a=False,quant_a_type='per_layer'):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.bitsw = num_bits_w
        self.bitsa = num_bits_a
        
        self.quant_w = quant_w
        self.quant_a = quant_a
        self.quant_a_type = quant_a_type
        assert self.quant_a_type in ['per_layer','per_channel']

        self.set_agregation(self.kernel_size[0]*self.kernel_size[1])

        self.alpha = nn.Parameter(torch.tensor(3.0),requires_grad=False)
        self.register_buffer('alpha_init', torch.zeros(1))
        self.bias_corr = False
        self.per_layer=False

        self.quant_basis = False
        self.bitsb = 8
        self.bit_alloc_w = False
        self.map_bitsw = torch.ones(self.out_channels)*self.bitsw
        self.bit_alloc_a = False
        self.map_bitsa = torch.ones(self.in_channels)*self.bitsa
        self.quantile = 0.9997
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
        return(self.d[n])
    
    def compute_bit_alloc_policy_w(self): #weight bit allocation
        if self.bit_alloc_w:
            B = self.out_channels*self.bitsw
            Bw = self.out_channels*2**self.bitsw #budget for weights
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
            self.map_bitsw = torch.ones(self.out_channels)*self.bitsw #same bitwidth for all channels
    
    def compute_bit_alloc_policy_a(self,alphas): #activation bit allocation
        if self.bit_alloc_a:
            B = torch.numel(alphas)*self.bitsa
            Ba = torch.numel(alphas)*2**self.bitsa #budget for activations
            alphas = alphas**(2/3)
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
            while B < self.map_bitsa.sum() and it<50:
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
            self.map_bitsa = torch.ones(self.in_channels)*self.bitsa #same bitwidth for all channels
            self.map_bitsa = self.map_bitsa.unsqueeze(0).unsqueeze(2).unsqueeze(3)

    
    def calibrate_base_per_layer(self, steps, step_lengths,init_basis, learn=False): #basis calibration per layer
        
        cube = (self.kernel_size==(1,1))
        denom = torch.max(torch.abs(self.weight.detach()))
        filters_FP = self.weight / denom
        #print(filters_FP.shape)

        B = (torch.eye(self.agreg, dtype=torch.float)).cuda()
        B = B / ((2**(self.bitsw-1))-1)
        minloss = loss_torch_linear(self.bitsw,filters_FP, B, self.bias_corr and self.bitsw<4).detach() # and self.bitsw<4

        for j in range(len(steps)):
            T = steps[j]

            for k in range(step_lengths[j]):
                testbasis = transform_ball_torch(B,T/2,cube)
                if self.quant_basis:
                    quantize_basis_layer(testbasis, self.bitsb)
                l = loss_torch_linear(self.bitsw,filters_FP, testbasis, self.bias_corr and self.bitsw<4).detach()  #and self.bitsw<4
                if l<minloss:
                    minloss=l
                    B = testbasis
        print("MCE = " + str(minloss.mean().item()))
        
        if init_basis: #first restart
            self.base = nn.Parameter(B,requires_grad=False) # Turn requires_grad to False not to learn the base
        else: #other restarts
            if minloss < loss_torch_linear(self.bitsw,filters_FP, self.base, self.bias_corr and self.bitsw<4).detach():
                self.base = nn.Parameter(B,requires_grad = False)




    def calibrate_base_per_channel(self, steps, step_lengths,init_basis, learn=False): #Basis calibration per channel

        cube = (self.kernel_size==(1,1))
        denom = torch.max(torch.abs(self.weight.detach()))
        filters_FP = self.weight / denom

        B = (torch.eye(self.agreg, dtype=torch.float)*torch.ones((filters_FP.shape[0],self.agreg,self.agreg))).cuda()
        B = 10**(-4) * B / (2**(self.bitsw-1))

        score1 = loss_torch(self.map_bitsw,filters_FP, B, self.bias_corr and self.bitsw<4).detach()
        
        
        if init_basis:
            self.compute_bit_alloc_policy_w()

        #print(self.map_bitsw)
        for j in range(len(steps)):
            T = steps[j]
            for k in range(step_lengths[j]):
                testbasis = transform_ball_torch(B,T/2,cube)
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

        if init_basis: #1st restart
            print("MCE = " + str(score1.mean().item()))
            self.base = nn.Parameter(B,requires_grad=False)
        else: #other restarts
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
        
    
    def bias_correction(self,weight):
        FP_w = self.weight.detach()
        w = weight.detach()
        ones = torch.ones_like(w)
        mu = (FP_w-w).mean(dim=(1,2,3),keepdim=True)*ones
        eps = 10**(-6)
        d = torch.linalg.norm((w - w.mean(dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)),dim=(1,2,3))+eps
        inv = 1/d
        inv[inv.isnan()]=0
        eta = torch.linalg.norm((FP_w - FP_w.mean(dim=(1,2,3)).unsqueeze(1).unsqueeze(2).unsqueeze(3)),dim=(1,2,3))*inv
        w = eta.unsqueeze(1).unsqueeze(2).unsqueeze(3)*(w+mu)
        return(w)

    
    def forward(self, inp):
        if self.quant_w: #weight quantization
            ### Weights in range [-1,1]
            denom = torch.max(torch.abs(self.weight.detach()))
            weight = self.weight / denom


            weight = weight.reshape(weight.shape[0],-1,self.agreg)
            basis = self.base
            # Ortho process with gram_schmidt
            if self.per_layer:
                ortho_B = gram_schmidt_torch(basis)
                # Find closest vector with babai
                coord = babai_torch(self.bitsw, basis, ortho_B, weight) #quantized parameters
            else:
                ortho_B = gram_schmidt_torch_channel(basis)
                # Find closest vector with babai
                coord = babai_torch_channel(self.map_bitsw, basis, ortho_B, weight) #quantized parameters

            #check quantization (lower bound)
            assert(torch.all(torch.le(torch.floor(-2**(self.map_bitsw-1)).cuda()+1,coord.flatten(1).min(dim=1)[0])).item()), "lower bound exceeded"
            #check quantization (upper bound)
            assert(torch.all(torch.le(coord.flatten(1).max(dim=1)[0],torch.floor(2**(self.map_bitsw-1)).cuda())).item()), "upper bound exceeded"

            weight = coordinates_to_vect_torch(basis,coord)
            weight = weight.reshape_as(self.weight)

            weight = weight*denom

            if self.bias_corr:
                weight = self.bias_correction(weight)
            
            
        else:
            ### FP weights
            weight = self.weight

        '''
        Activation quantization
        '''
        if self.quant_a: #activations quantization
            input_val = inp
            if self.alpha_init == 0: #thresholds initialization
                
                if self.quant_a_type=='per_layer':

                    alpha = input_val.abs().max() #very naive
                    self.map_bitsa = (torch.ones(self.in_channels)*self.bitsa).unsqueeze(1).unsqueeze(2).unsqueeze(0)

                elif self.quant_a_type=='per_channel':

                    alpha = input_val.abs().transpose(0,1).flatten(1) #get activations per channel
                    quantile = self.quantile #temporary quantile
                    alpha = torch.quantile(alpha,quantile,dim=1,keepdim=True) #per channel quantile
                    alpha = alpha.transpose(0,1).unsqueeze(2).unsqueeze(3) #reshape alpha
                    self.compute_bit_alloc_policy_a(alpha)
                    quantile = self.map_bitsa.flatten().clone().cpu().apply_(self.broadcast_quantiles).cuda() #quantile per channel
                    alpha = input_val.transpose(0,1).flatten(1) #get activations per channel
                    alpha = torch.diagonal(torch.quantile(alpha,quantile,dim=1)).unsqueeze(1) #per channel quantile
                    alpha = alpha.transpose(0,1).unsqueeze(2).unsqueeze(3) #reshape alpha
                    
                alpha[alpha==0] = 1 #arbitrary value when quantile is 0
                self.alpha = torch.nn.Parameter(alpha,requires_grad=False)
                self.alpha_init.fill_(1)

            if input_val.min()<0:
                input_val = torch.clamp(input_val/self.alpha,-1,1)
                input_val = (input_val+1) /2
            else:
                input_val = torch.clamp(input_val/self.alpha,0,1)
            bins = (2**self.map_bitsa-1).cuda()
            input_val = 1/bins * (torch.round(input_val*bins))

            if input_val.min()<0:
                input_val = 2*input_val - 1

            input_val = input_val*(self.alpha)

        else:
            '''
            Activation Full precision
            '''
            input_val = inp
        output = F.conv2d(input_val, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        return output