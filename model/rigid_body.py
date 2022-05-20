import torch
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



def skew(w):

    zero = torch.zeros_like( w[...,0])
    W = torch.stack ( [ zero, -w[...,2], w[...,1],
                        w[...,2], zero, -w[...,0],
                        -w[...,1], w[...,0], zero], dim=-1).reshape((-1,3,3))
    return W

def exp_se3( w, v, theta):
    '''
    :param w:
    :param v:
    :param theta:
    :return:
    '''

    theta=theta[...,None]
    W = skew(w)
    I= torch.eye(3)[None].to(theta)
    R = I + torch.sin(theta) * W +  (1-torch.cos(theta)) * W @ W
    p = I + (1-torch.cos(theta) ) * W + (theta - torch.sin(theta)) * W @ W
    t = p @ v[...,None]
    return R, t

def exp_so3( w, theta):

    theta=theta[...,None]
    W = skew(w)
    I= torch.eye(3)[None].to(theta)
    R = I + torch.sin(theta) * W +  (1-torch.cos(theta)) * W @ W
    return R


