import torch
import torch.nn as nn
import torch.nn.functional as F

def _6d_to_SO3(d6):
    '''
    On the Continuity of Rotation Representations in Neural Networks, CVPR'19. c.f. http://arxiv.org/abs/1812.07035
    :param d6: [n, 6]
    :return: [n, 3, 3]
    '''
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def euler_to_SO3(euler_angles, convention = ['X', 'Y', 'Z']):
    '''
    :param euler_angles: [n, 6]
    :param convention: order of axis
    :return:
    '''

    def _axis_angle_rotation(axis, angle):
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        one = torch.ones_like(angle)
        zero = torch.zeros_like(angle)
        if axis == "X":
            R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
        elif axis == "Y":
            R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
        elif axis == "Z":
            R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
        else:
            raise ValueError("letter must be either X, Y or Z.")
        return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]

    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def quaternion_to_SO3(quaternions):
    '''
    :param quaternions: [n, 4]
    :return:
    '''

    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def exp_so3(w, theta):
    '''
    Rodrigues 公式的 batched 实现。
    输入:
      - w: (..., 3) 旋转轴（不要求单位长度）
      - theta: (..., 1) 或 (...,) 旋转角
    输出:
      - R: (..., 3, 3) 旋转矩阵
    '''
    # 统一 theta 形状为 (...,)
    if theta.dim() == w.dim():
        theta = theta[..., 0]

    # 安全单位化 w，避免 0 角导致的 NaN
    w_norm = torch.norm(w, dim=-1, keepdim=True)
    w_unit = torch.where(w_norm > 0, w / w_norm, torch.zeros_like(w))

    wx, wy, wz = w_unit[..., 0], w_unit[..., 1], w_unit[..., 2]
    zeros = torch.zeros_like(wx)
    K = torch.stack((
        zeros, -wz,   wy,
          wz, zeros, -wx,
         -wy,   wx, zeros
    ), dim=-1).reshape(w_unit.shape[:-1] + (3, 3))

    I = torch.eye(3, dtype=w.dtype, device=w.device).expand(K.shape)
    sin_t = torch.sin(theta)[..., None, None]
    one_minus_cos_t = (1.0 - torch.cos(theta))[..., None, None]

    return I + sin_t * K + one_minus_cos_t * (K @ K)

class Deformation_Pyramid ():

    def __init__(self, depth, width, device, k0, m, rotation_format, nonrigidity_est=False, motion='SE3'):

        pyramid = []


        assert motion in [ "Sim3", "SE3", "sflow"]


        for i in range (m):
            pyramid.append(
                NDPLayer(depth,
                         width,
                         k0,
                         i+1,
                         rotation_format,
                         nonrigidity_est=nonrigidity_est,
                         motion=motion
                         ).to(device)
            )


        self.pyramid = pyramid
        self.n_hierarchy = m

    def warp(self, x, max_level=None, min_level=0):

        if max_level is None:
            max_level = self.n_hierarchy - 1

        assert max_level < self.n_hierarchy, "more level than defined"

        data = {}

        for i in range(min_level, max_level + 1):
            x, nonrigidity = self.pyramid[i](x)
            data[i] = (x, nonrigidity)
        return x, data

    def gradient_setup(self, optimized_level):

        assert optimized_level < self.n_hierarchy, "more level than defined"

        # optimize current level, freeze the other levels
        for i in range( self.n_hierarchy):
            net = self.pyramid[i]
            if i == optimized_level:
                for param in net.parameters():
                    param.requires_grad = True
            else:
                for param in net.parameters():
                    param.requires_grad = False



class NDPLayer(nn.Module):
    def __init__(self, depth, width, k0, m, rotation_format="euler", nonrigidity_est=False, motion='SE3'):
        super().__init__()

        self.k0 = k0
        self.m = m
        dim_x = 6
        #dim_x = self.m * 6  # 修复：编码维度应该是 m * 6
        self.nonrigidity_est = nonrigidity_est
        self.motion = motion
        self.input= nn.Sequential( nn.Linear(dim_x,width), nn.ReLU())
        self.mlp = MLP(depth=depth,width=width)

        self.rotation_format = rotation_format


        """rotation branch"""
        if self.motion in [ "Sim3", "SE3"] :

            if self.rotation_format in [ "axis_angle", "euler" ]:
                self.rot_brach = nn.Linear(width, 3)
            elif self.rotation_format == "quaternion":
                self.rot_brach = nn.Linear(width, 4)
            elif self.rotation_format == "6D":
                self.rot_brach = nn.Linear(width, 6)


            if self.motion == "Sim3":
                self.s_branch = nn.Linear(width, 1) # scale branch


        """translation branch"""
        self.trn_branch = nn.Linear(width, 3)


        """rigidity branch"""
        if self.nonrigidity_est:
            self.nr_branch = nn.Linear(width, 1)
            self.sigmoid = nn.Sigmoid()


        # Apply small scaling on the MLP output, s.t. the optimization can start from near identity pose
        self.mlp_scale = 0.01

        self._reset_parameters()

    def forward (self, x):

        fea = self.posenc( x )
        fea = self.input(fea)
        fea = self.mlp(fea)

        t = self.mlp_scale * self.trn_branch ( fea )

        if self.motion == "SE3":
            R = self.get_Rotation(fea)
            x_ = (R @ x[..., None]).squeeze() + t

        elif self.motion == "Sim3":
            R = self.get_Rotation(fea)
            s = self.mlp_scale * self.s_branch(fea) + 1  # optimization starts with scale==1
            x_ = s * (R @ x[..., None]).squeeze() + t

        else: # scene flow
            x_ = x + t


        if self.nonrigidity_est:
            nonrigidity =self.sigmoid( self.mlp_scale * self.nr_branch(fea) )
            x_ = x + nonrigidity * (x_ - x)
            nonrigidity = nonrigidity.squeeze()
        else:
            nonrigidity = None


        return x_.squeeze(), nonrigidity



    def get_Rotation (self, fea):

        R = self.mlp_scale * self.rot_brach( fea )

        if self.rotation_format == "euler":
            R = euler_to_SO3(R)
        elif self.rotation_format == "axis_angle":
            theta = torch.norm(R, dim=-1, keepdim=True)
            w = R / theta
            R = exp_so3(w, theta)
        elif self.rotation_format =='quaternion':
            s = (R * R).sum(1)
            R = R / _copysign(torch.sqrt(s), R[:, 0])[:, None]
            R = quaternion_to_SO3(R)
        elif self.rotation_format == "6D":
            R = _6d_to_SO3(R)

        return R


    def posenc(self, pos):
        pi = 3.14
        x_position, y_position, z_position = pos[..., 0:1], pos[..., 1:2], pos[..., 2:3]
        #mul_term = ( 2 ** (torch.arange(self.m, device=pos.device).float() + self.k0) * pi ).reshape(1, -1)
        mul_term = (2 ** (self.m + self.k0)  )#.reshape(1, -1)

        sinx = torch.sin(x_position * mul_term)
        cosx = torch.cos(x_position * mul_term)
        siny = torch.sin(y_position * mul_term)
        cosy = torch.cos(y_position * mul_term)
        sinz = torch.sin(z_position * mul_term)
        cosz = torch.cos(z_position * mul_term)
        pe = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=-1)
        # print(pe.shape)

        return pe


    def _reset_parameters(self):
        # 原始初始化方法：所有参数都使用xavier_uniform_初始化
        # 这会导致nonrigidity分支产生0-1之间的随机初始值
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)        




class MLP(torch.nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        self.pts_linears = nn.ModuleList( [nn.Linear(width, width) for i in range(depth - 1)])

    def forward(self, x):
        for i, l in enumerate(self.pts_linears):
            x = self.pts_linears[i](x)
            x = F.relu(x)
        return x
