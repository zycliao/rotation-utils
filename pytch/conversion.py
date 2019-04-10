import torch

EPS = 1e-8


def expmap2rotmat(r):
    """
    :param r: Axis-angle, Nx3
    :return: Rotation matrix, Nx3x3
    """
    dev = r.device
    assert r.shape[1] == 3
    bs = r.shape[0]
    theta = torch.sqrt(torch.sum(torch.pow(r, 2), 1, keepdim=True))
    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    eye = torch.unsqueeze(torch.eye(3), 0).repeat(bs, 1, 1).to(dev)
    norm_r = r / (theta + EPS)
    r_1 = torch.unsqueeze(norm_r, 2)  # N, 3, 1
    r_2 = torch.unsqueeze(norm_r, 1)  # N, 1, 3
    zero_col = torch.zeros(bs, 1).to(dev)
    skew_sym = torch.cat([zero_col, -norm_r[:, 2:3], norm_r[:, 1:2], norm_r[:, 2:3], zero_col,
                          -norm_r[:, 0:1], -norm_r[:, 1:2], norm_r[:, 0:1], zero_col], 1)
    skew_sym = skew_sym.contiguous().view(bs, 3, 3)
    R = cos_theta*eye + (1-cos_theta)*torch.bmm(r_1, r_2) + sin_theta*skew_sym
    return R


def rotmat2expmap(R):
    """
    :param R: Rotation matrix, Nx3x3
    :return: r: Rotation vector, Nx3
    """
    assert R.shape[1] == R.shape[2] == 3
    theta = torch.acos(torch.clamp((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, min=-1., max=1.)).view(-1, 1)
    r = torch.stack((R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]), 1) / (2*torch.sin(theta))
    r_norm = r / torch.sqrt(torch.sum(torch.pow(r, 2), 1, keepdim=True))
    return theta * r_norm


def quat2expmap(q):
    """
    :param q: quaternion, Nx4
    :return: r: Axis-angle, Nx3
    """
    assert q.shape[1] == 4
    cos_theta_2 = torch.clamp(q[:, 0: 1], min=-1., max=1.)
    theta = torch.acos(cos_theta_2)*2
    sin_theta_2 = torch.sqrt(1-torch.pow(cos_theta_2, 2))
    r = theta * q[:, 1:4] / (sin_theta_2 + EPS)
    return r


def expmap2quat(r):
    """
    :param r: Axis-angle, Nx3
    :return: q: quaternion, Nx4
        """
    assert r.shape[1] == 3
    theta = torch.sqrt(torch.sum(torch.pow(r, 2), 1, keepdim=True))
    unit_r = r / theta
    theta_2 = theta / 2.
    cos_theta_2 = torch.cos(theta_2)
    sin_theta_2 = torch.sin(theta_2)
    q = torch.cat((cos_theta_2, unit_r*sin_theta_2), 1)
    return q
