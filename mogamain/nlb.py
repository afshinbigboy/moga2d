

import torch  
import torch.nn as nn  
import torch.nn.functional as F  


class WeightedNonLocalBlock(nn.Module):  
    def __init__(self, input_channels):  
        super(WeightedNonLocalBlock, self).__init__()  

        # Point-wise convolutions for theta, phi, and g - maintaining input channels  
        self.theta_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)  
        self.phi_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)  
        self.g_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)  

        # Learnable weight (scalar)  
        self.w = nn.Parameter(torch.tensor(0.5))  # Initial weight  

        # Final point-wise convolution for output feature map  
        self.Wz_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)  

    def forward(self, x):  
        # Step 1: Compute theta, phi, and g using convolutions  
        theta_x = self.theta_conv(x)  # (N, C, H, W)  
        phi_x = self.phi_conv(x)      # (N, C, H, W)  
        g_x = self.g_conv(x)          # (N, C, H, W)  

        # Step 2: Reshape tensors to prepare for attention computation  
        batch_size, channels, height, width = x.size()  
        
        theta_x = theta_x.view(batch_size, channels, -1)  # (N, C, H*W)  
        phi_x = phi_x.view(batch_size, channels, -1)      # (N, C, H*W)  
        g_x = g_x.view(batch_size, channels, -1)          # (N, C, H*W)  

        # Step 3: Compute the attention map  
        attention_map = F.softmax(torch.bmm(theta_x.transpose(1, 2), phi_x), dim=-1)  # (N, H*W, H*W)  

        # Step 4: Compute the non-local operation  
        y = torch.bmm(attention_map, g_x.transpose(1, 2))  # (N, H*W, C)  

        # Step 5: Reshape y back to spatial dimensions  
        y = y.view(batch_size, channels, height, width)  # (N, C, H, W)  

        # Step 6: Combine the output with the original input  
        z = (1 - self.w) * x + self.w * self.Wz_conv(y)  # Apply final convolution  

        return z  

# Example usage  
if __name__ == "__main__":  
    input_tensor = torch.randn(8, 64, 32, 32)  # Example input: (batch_size, channels, height, width)  
    model = WeightedNonLocalBlock(input_channels=64)  
    output = model(input_tensor)  
    print(output.shape)  # Check output shape

