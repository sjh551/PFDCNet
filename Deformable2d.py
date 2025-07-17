import torch
import torch.nn as nn
# from ops_dcnv3.modules import dcnv3
import torch
import torch.nn as nn
import torch.nn.functional as F
from CBAM import cbam
import torch.nn.init as init
from offset_to_center import get_cam_center,compute_loss

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=True):
        """
        把每个像素点的可变形卷积需要获取的数给搞出来，拼接成更大的图像，然后步幅就是kernel_size
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
            modulation 表示 DCNV1还是DCNV2的开关
        """
        super(DeformConv2d, self).__init__()
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)
        self.inc = inc
        self.cbam = cbam(in_channel=inc)
        # 用于获取偏置项 卷积核的通道数应该为2xkernel_sizexkernel_size
        # 结果为[b,2xkernel_sizexkernel_size,h,w]
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        # 将权重用0来进行填充
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_full_backward_hook(self._set_lr)   # 注册为反向传播的钩子，反向传播时调用

        self.kernel_size = kernel_size
        self.modulation = modulation 

        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_full_backward_hook(self._set_lr)
        

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        # grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        # grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))
        grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return grad_input

    def init_weights(self):
        """自定义初始化方法"""
        print("init successfully!")

        # 对偏移量的初始化，通常希望初始偏移为0
        init.constant_(self.p_conv.weight, 0)
        init.constant_(self.p_conv.bias, 0)

        # 对主卷积层 self.conv 进行初始化
        init.kaiming_uniform_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)

        # 如果使用 modulation（DCNv2），需要初始化 self.m_conv
        if self.modulation:
            init.constant_(self.m_conv.weight, 0)
            init.constant_(self.m_conv.bias, 0)


    def forward(self, x,centers):
        device = x.device
        b, c, _ , _ = x.size()
        attn = self.cbam(x)
        offset = self.p_conv(x)     # 生成偏移
        # print(offset)

        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))   # 乘以调制

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2
        # 原来输入为[1,3,32,32]---->[1,3,34,34]
        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)   # 这样就能得到每个像素位置下每个卷积核位置应该采样的位置，即采样坐标
        
        # centers = get_cam_center(attn)       # 计算中心坐标
        loss = compute_loss(p,centers)      # 得到每个采样位置的损失
        if torch.isnan(loss):
            print(f'centers:{centers}')



        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        # print(x_offset.shape) 
        out = self.conv(x_offset)
        
        return out,loss
        


    def _get_p_n(self, N, dtype):
        # 生成卷积核的矩阵
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),  # [-1,1]
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))  # [-1,1]
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        # 以卷积核中心为原点，然后生成相对坐标
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        """
        用来把全局图像编号
        :param h:
        :param w:
        :param N:
        :param dtype:
        :return:
        """
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)  # 相当于在第二个维度上重复N次
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)   # N为通道数,且为卷积核的size x size

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        # print(p.shape)
        return p

    def _get_x_q(self, x, q, N):
        """
        根据 q 中存储的坐标信息，从图像 x 中提取特定位置的像素值，生成一个新的张量 x_offset，q表示左上、右下、左下、右上角信息
        其形状为 (b, c, h, w, N)，表示经过形变的图像特征。
        :param x:
        :param q:
        :param N:
        :return:
        """
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset



class GroupDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride,padding, groups):
        """
        inc:输入通道数
        outc:输出通道数
        groups: how many groups need
        
        """
        super(GroupDeformConv, self).__init__()
        
        self.groups = groups
        self.in_channels_per_group = in_channels // groups
        
        self.deform_convs = nn.ModuleList()
        self.cbam_cams = nn.ModuleList()
        for i in range(groups):
            # 为每组定义cbam获取热力图
            self.cbam_cams.append(
                cbam(self.in_channels_per_group)
            )
            # 为每个组定义一个DeformConv2d卷积
            self.deform_convs.append(
                DeformConv2d(
                    self.in_channels_per_group, 
                    out_channels // groups,  # 假设输出通道数也要分组
                    kernel_size=kernel_size, 
                    stride=stride, 
                    padding=padding
                )
            )
        

    def forward(self, x):
        # 将输入特征图沿着通道维度分成多个组
        split_x = torch.split(x, self.in_channels_per_group, dim=1)
        # print(self.in_channels_per_group)
        total_loss = None
        # 对每个组应用DeformConv2d
        outputs = None

        for i in range(self.groups):
            cam = self.cbam_cams[i](split_x[i])     # 计算每组的热力图
            output,loss = self.deform_convs[i](split_x[i],cam)
            if outputs is None:
                outputs=output
            else:
                outputs = torch.cat((outputs,output),dim=1)

            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss
            # print(outputs.shape)
        return outputs,total_loss
    
    
if __name__ == "__main__":
    x = torch.randn(2,3,224,224)
    model = DeformConv2d(inc=3,outc=3,kernel_size=1,padding=0)
    y,loss = model(x)
    