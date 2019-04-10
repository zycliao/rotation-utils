import numpy as np

EPS = 1e-8


def expmap2rotmat(r):
    """
    :param r: Axis-angle, Nx3
    :return: Rotation matrix, Nx3x3
    """
    assert r.shape[1] == 3
    bs = r.shape[0]
    theta = np.sqrt(np.sum(np.square(r), 1, keepdims=True))
    cos_theta = np.expand_dims(np.cos(theta), -1)
    sin_theta = np.expand_dims(np.sin(theta), -1)
    eye = np.tile(np.expand_dims(np.eye(3), 0), (bs, 1, 1))
    norm_r = r / (theta + EPS)
    r_1 = np.expand_dims(norm_r, 2)  # N, 3, 1
    r_2 = np.expand_dims(norm_r, 1)  # N, 1, 3
    zero_col = np.zeros([bs, 1]).astype(r.dtype)
    skew_sym = np.concatenate([zero_col, -norm_r[:, 2:3], norm_r[:, 1:2], norm_r[:, 2:3], zero_col,
                          -norm_r[:, 0:1], -norm_r[:, 1:2], norm_r[:, 0:1], zero_col], 1)
    skew_sym = skew_sym.reshape(bs, 3, 3)
    R = cos_theta*eye + (1-cos_theta)*np.einsum('npq,nqu->npu', r_1, r_2) + sin_theta*skew_sym
    return R


def rotmat2expmap(R):
    """
    :param R: Rotation matrix, Nx3x3
    :return: r: Rotation vector, Nx3
    """
    assert R.shape[1] == R.shape[2] == 3
    theta = np.arccos(np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, -1., 1.)).reshape([-1, 1])
    r = np.stack((R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]), 1) / (2*np.sin(theta))
    r_norm = r / np.sqrt(np.sum(np.square(r), 1, keepdims=True))
    return theta * r_norm


def quat2expmap(q):
    """
    :param q: quaternion, Nx4
    :return: r: Axis-angle, Nx3
    """
    assert q.shape[1] == 4
    cos_theta_2 = np.clip(q[:, 0: 1], -1., 1.)
    theta = np.arccos(cos_theta_2)*2
    sin_theta_2 = np.sqrt(1-np.square(cos_theta_2))
    r = theta * q[:, 1:4] / (sin_theta_2 + EPS)
    return r


def expmap2quat(r):
    """
    :param r: Axis-angle, Nx3
    :return: q: quaternion, Nx4
        """
    assert r.shape[1] == 3
    theta = np.sqrt(np.sum(np.square(r), 1, keepdims=True))
    unit_r = r / theta
    theta_2 = theta / 2.
    cos_theta_2 = np.cos(theta_2)
    sin_theta_2 = np.sin(theta_2)
    q = np.concatenate((cos_theta_2, unit_r*sin_theta_2), 1)
    return q
