from quantize_filters import *

#File for quantized pooling layers


class QAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d): #quantized adaptive average pooling layer
    def __init__(self, output_size, quant_a=False, num_bits_a=8, quant_a_type='per_layer'):
        super(nn.AdaptiveAvgPool2d,self).__init__(output_size)
        self.quant_a = quant_a
        self.bitsa = num_bits_a
        self.register_buffer('alpha_init', torch.zeros(1))
        self.quant_a_type = quant_a_type
        assert(self.quant_a_type in ['per_layer','per_channel'])
    
    def set_bits(self,bits_a):
        self.bitsa = bits_a

    def forward(self,inp):
        
        input_val = inp
        
        if self.quant_a: #activations quantization
            if self.alpha_init == 0:
                alpha = input_val.abs().max() #naive per layer activation quantization
                self.alpha = torch.nn.Parameter(alpha,requires_grad=False)
                self.alpha_init.fill_(1)

            if input_val.min()<0: #signed activations
                input_val = torch.clamp(input_val/self.alpha,-1,1)
                input_val = (input_val+1) /2
            else:
                input_val = torch.clamp(input_val/self.alpha,0,1)
            
            bins = (2**self.bitsa-1)
            input_val = 1/bins * (torch.round(input_val*bins))

            if input_val.min()<0:
                input_val = 2*input_val - 1

            input_val = input_val*(self.alpha)

        m = F.adaptive_avg_pool2d(input_val,self.output_size)
        return m

class QMaxPool2d(nn.MaxPool2d): #quantized max pooling layer
    def __init__(self, kernel_size, stride, padding, dilation, ceil_mode, quant_a = False, num_bits_a=8,quant_a_type='per_layer'):
        super(nn.MaxPool2d,self).__init__(kernel_size, stride, padding, dilation, ceil_mode)
        
        self.quant_a = quant_a
        self.bitsa = num_bits_a
        self.register_buffer('alpha_init', torch.zeros(1))
        self.quant_a_type = quant_a_type
        assert(self.quant_a_type in ['per_layer','per_channel'])
    
    def set_bits(self,bits_a):
        self.bitsa = bits_a

    def forward(self,inp):
        
        input_val = inp
        
        if self.quant_a: #activations quantization
            if self.alpha_init == 0:
                alpha = input_val.abs().max() #naive per layer activation quantization
                self.alpha = torch.nn.Parameter(alpha,requires_grad=False)
                self.alpha_init.fill_(1)

            if input_val.min()<0: #signed activations
                input_val = torch.clamp(input_val/self.alpha,-1,1)
                input_val = (input_val+1) /2
            else:
                input_val = torch.clamp(input_val/self.alpha,0,1)
            
            bins = (2**self.bitsa-1)
            input_val = 1/bins * (torch.round(input_val*bins))

            if input_val.min()<0:
                input_val = 2*input_val - 1

            input_val = input_val*(self.alpha)

        m = F.max_pool2d(input_val,kernel_size=self.kernel_size,stride=self.stride,padding=self.padding,dilation=self.dilation,ceil_mode=self.ceil_mode)
        return m