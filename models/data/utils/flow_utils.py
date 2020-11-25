import numpy as np
import os.path
from PIL import Image
import cv2
import re

TAG_CHAR = np.array([202021.25], np.float32)
UNKNOWN_FLOW_THRESH = 1e7

def read_flow(filename: str):
    """ 
    Read .flo file in Middlebury format
    Args:
        filename: str  file name
    Return:
        optical flow: numpy.ndarray [h, w, 2]
    """
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)

            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))
        
def save_flow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Args:
        filename: str file name
        uv: numpy.ndarray [h, w, 2] if v is None, else [h, w]
        v: numpy.ndarray [h, w] or None
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    with open(filename,'wb') as f:
        # write the header
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width*nBands))
        tmp[:,np.arange(width)*2] = u
        tmp[:,np.arange(width)*2 + 1] = v
        tmp.astype(np.float32).tofile(f)
        
def save_flow_as_image(filename, save_dir):
    flow = read_flow(filename)
    img = flow2img(flow)
    
    idx = filename.rfind('/') + 1
    plt.imsave(os.path.join(save_dir, '{}-vis.png'.format(filename[idx:-4])), img)

def read_kitti_png_file(flow_file):
    flow_img = cv2.imread(flow_file, -1)
    flow_img = flow_img.astype(np.float32)
    flow_data = np.zeros(flow_img.shape, dtype=np.float32)
    flow_data[:, :, 0] = (flow_img[:, :, 2] - 2**15) / 64.0
    flow_data[:, :, 1] = (flow_img[:, :, 1] - 2**15) / 64.0
    flow_data[:, :, 2] = flow_img[:, :, 0]
    return flow_data


def read_pfm_file(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data  #, scale


def resize_flow(flow, des_height, des_width, method='bilinear'):
    # improper for sparse flow
    src_height = flow.shape[0]
    src_width = flow.shape[1]
    if src_width == des_width and src_height == des_height:
        return flow
    ratio_height = float(des_height) / float(src_height)
    ratio_width = float(des_width) / float(src_width)
    if method == 'bilinear':
        flow = cv2.resize(
            flow, (des_width, des_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'nearest':
        flow = cv2.resize(
            flow, (des_width, des_height), interpolation=cv2.INTER_NEAREST)
    else:
        raise Exception('Invalid resize flow method!')
    flow[:, :, 0] = flow[:, :, 0] * ratio_width
    flow[:, :, 1] = flow[:, :, 1] * ratio_height
    return flow    
    
def flow2img(flow):
    """
    convert optical flow into color image
    Args:
        flow: np.ndarray [h, w, 2]
    Returns:
        img: np.ndarray [h, w, 3]
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    
    UNKNOW_FLOW_THRESHOLD = 1e7
    pr1 = abs(u) > UNKNOW_FLOW_THRESHOLD
    pr2 = abs(v) > UNKNOW_FLOW_THRESHOLD
    idx_unknown = (pr1 | pr2)
    u[idx_unknown] = v[idx_unknown] = 0

    # get max value in each direction
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxu = max(maxu, np.max(u))
    maxv = max(maxv, np.max(v))
    minu = min(minu, np.min(u))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))
    u = u / maxrad + np.finfo(float).eps
    v = v / maxrad + np.finfo(float).eps

    img = compute_color(u, v)

    idx = np.repeat(idx_unknown[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def flow_error(tu, tv, u, v):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (
        abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = ((np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow))
    index_su = su[ind2]
    index_sv = sv[ind2]
    an = 1.0 / np.sqrt(index_su**2 + index_sv**2 + 1)

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    tn = 1.0 / np.sqrt(index_stu**2 + index_stv**2 + 1)
    '''
    angle = un * tun + vn * tvn + (an * tn)
    index = [angle == 1.0]
    angle[index] = 0.999
    ang = np.arccos(angle)
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    '''

    epe = np.sqrt((stu - su)**2 + (stv - sv)**2)
    epe = epe[ind2]
    mepe = np.mean(epe)
    return mepe

def flow_kitti_error(tu, tv, u, v, mask):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :param mask: ground-truth mask
    :return: End point error of the estimated flow
    """
    tau = [3, 0.05]
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]
    smask = mask[:]

    ind_valid = (smask != 0)
    n_total = np.sum(ind_valid)

    epe = np.sqrt((stu - su)**2 + (stv - sv)**2)
    mag = np.sqrt(stu**2 + stv**2) + 1e-5

    epe = epe[ind_valid]
    mag = mag[ind_valid]

    err = np.logical_and((epe > tau[0]), (epe / mag) > tau[1])
    n_err = np.sum(err)

    mean_epe = np.mean(epe)
    mean_acc = 1 - (float(n_err) / float(n_total))
    return (mean_epe, mean_acc)

def evaluate_flow_file(gt_file, pred_file):
    """
    evaluate the estimated optical flow end point error according to ground truth provided
    :param gt_file: ground truth file path
    :param pred_file: estimated optical flow file path
    :return: end point error, float32
    """
    # Read flow files and calculate the errors
    gt_flow = read_flow(gt_file)  # ground truth flow
    eva_flow = read_flow(pred_file)  # predicted flow
    # Calculate errors
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1],
                            eva_flow[:, :, 0], eva_flow[:, :, 1])
    return average_pe


def evaluate_flow(gt_flow, pred_flow):
    """
    gt: ground-truth flow
    pred: estimated flow
    """
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1],
                            pred_flow[:, :, 0], pred_flow[:, :, 1])
    return average_pe


def evaluate_kitti_flow(gt_flow, pred_flow, rigid_flow=None):
    if gt_flow.shape[2] == 2:
        gt_mask = np.ones((gt_flow.shape[0], gt_flow.shape[1]))
        epe, acc = flow_kitti_error(gt_flow[:, :, 0], gt_flow[:, :, 1],
                                    pred_flow[:, :, 0], pred_flow[:, :, 1],
                                    gt_mask)
    elif gt_flow.shape[2] == 3:
        epe, acc = flow_kitti_error(gt_flow[:, :, 0], gt_flow[:, :, 1],
                                    pred_flow[:, :, 0], pred_flow[:, :, 1],
                                    gt_flow[:, :, 2])
    return (epe, acc)

def horizontal_flip_flow(flow):
    flow = np.copy(np.fliplr(flow))
    flow[:, :, 0] *= -1
    return flow


def vertical_flip_flow(flow):
    flow = np.copy(np.flipud(flow))
    flow[:, :, 1] *= -1
    return flow


def remove_ambiguity_flow(flow_img, err_img, threshold_err=10.0):
    thre_flow = flow_img
    mask_img = np.ones(err_img.shape, dtype=np.uint8)
    mask_img[err_img > threshold_err] = 0.0
    thre_flow[err_img > threshold_err] = 0.0
    return (thre_flow, mask_img)


def write_kitti_png_file(flow_fn, flow_data, mask_data):
    flow_img = np.zeros((flow_data.shape[0], flow_data.shape[1], 3),
                        dtype=np.uint16)
    flow_img[:, :, 2] = flow_data[:, :, 0] * 64.0 + 2**15
    flow_img[:, :, 1] = flow_data[:, :, 1] * 64.0 + 2**15
    flow_img[:, :, 0] = mask_data[:, :]
    cv2.imwrite(flow_fn, flow_img)


def flow_kitti_mask_error(tu, tv, gt_mask, u, v, pd_mask):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param gt_mask: ground-truth mask

    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :param pd_mask: estimated flow mask
    :return: End point error of the estimated flow
    """
    tau = [3, 0.05]
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]
    s_gt_mask = gt_mask[:]
    s_pd_mask = pd_mask[:]

    ind_valid = np.logical_and(s_gt_mask != 0, s_pd_mask != 0)
    n_total = np.sum(ind_valid)

    epe = np.sqrt((stu - su)**2 + (stv - sv)**2)
    mag = np.sqrt(stu**2 + stv**2) + 1e-5

    epe = epe[ind_valid]
    mag = mag[ind_valid]

    err = np.logical_and((epe > tau[0]), (epe / mag) > tau[1])
    n_err = np.sum(err)

    mean_epe = np.mean(epe)
    mean_acc = 1 - (float(n_err) / float(n_total))
    return (mean_epe, mean_acc)

def compute_color(u, v):
    """
    compute optical flow color map
    Args:
        u: horizontal optical flow, np.ndarray [h, w]
        v: vertical optical flow, np.ndarray [h, w]
    Return:
        img: converted image, np.ndarray [h, w, 3]
    """

    height, width = u.shape
    img = np.zeros((height, width, 3))

    NAN_idx = np.isnan(u) | np.isnan(v)
    u[NAN_idx] = v[NAN_idx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - NAN_idx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    Return: Color wheel, np.ndarray [ncols, 3]
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel