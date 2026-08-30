import numpy as np
import mrcfile
import argparse
import time
from numba import njit, prange

class CMD:
    def __init__(self):
        self.filename = ""
        self.file = ""
        self.Nthr = 2
        self.dreso = 16.0
        self.Mode = 0
        self.th1 = 0.0
        self.ssize = 7.0
        self.out_file = ""

class MRC:
    def __init__(self):
        self.filename = ""
        self.xdim = 0
        self.ydim = 0
        self.zdim = 0
        self.ncstart = 0
        self.nrstart = 0
        self.nsstart = 0
        self.mx = 0
        self.my = 0
        self.mz = 0
        self.xlen = 0.0
        self.ylen = 0.0
        self.zlen = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.gamma = 0.0
        self.mapc = 0
        self.mapr = 0
        self.maps = 0
        self.dmin = 0.0
        self.dmax = 0.0
        self.dmean = 0.0
        self.ispg = 0
        self.nsymbt = 0
        self.orgxyz = np.zeros(3)
        self.NumVoxels = 0
        self.dens = None
        self.vec = None
        self.xyz = None
        self.widthx = 0.0
        self.widthy = 0.0
        self.widthz = 0.0
        self.Nact = 0
        self.dmax2 = 0.0
        self.dsum = 0.0
        self.std = 0.0
        self.ave = 0.0
        self.cent = np.zeros(3)
        self.std_norm_ave = 0.0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", dest="file", required=True, help="MAP.mrc file")
    parser.add_argument("-t", dest="th1", type=float, default=0.0, help="Threshold of density map1")
    parser.add_argument("-g", dest="dreso", type=float, default=16.0, help="Bandwidth of the Gaussian filter")
    parser.add_argument("-s", dest="ssize", type=float, default=7.0, help="Sampling voxel spacing")
    parser.add_argument("-c", dest="Nthr", type=int, default=2, help="Number of cores for threads")
    parser.add_argument("-V", dest="Mode", action="store_true", help="Vector coordinate information")
    parser.add_argument("-o", dest="out_file", default="", help="Output file path for vector information")
    return parser.parse_args()


def readmrc(mrc: MRC, filename: str):
    with mrcfile.open(filename) as mrc_file:
        header = mrc_file.header

        mrc.xdim = int(header.nx)
        mrc.ydim = int(header.ny)
        mrc.zdim = int(header.nz)
        mrc.ncstart = int(header.nxstart)
        mrc.nrstart = int(header.nystart)
        mrc.nsstart = int(header.nzstart)
        mrc.mx = int(header.mx)
        mrc.my = int(header.my)
        mrc.mz = int(header.mz)
        mrc.widthx = float(header.cella.x) / int(header.nx)
        mrc.widthy = float(header.cella.y) / int(header.ny)
        mrc.widthz = float(header.cella.z) / int(header.nz)
        mrc.alpha, mrc.beta, mrc.gamma = header.cellb.alpha, header.cellb.beta, header.cellb.gamma
        mrc.mapc = int(header.mapc)
        mrc.mapr = int(header.mapr)
        mrc.maps = int(header.maps)
        mrc.dmin = float(header.dmin)
        mrc.dmax = float(header.dmax)
        mrc.dmean = float(header.dmean)
        mrc.ispg = int(header.ispg)
        mrc.nsymbt = int(header.nsymbt)
        try:
            mrc.orgxyz = np.array([float(header.origin[0]), float(header.origin[1]), float(header.origin[2])])
        except Exception:
            mrc.orgxyz = np.zeros(3)
        mrc.NumVoxels = mrc.xdim * mrc.ydim * mrc.zdim
        mrc.dens = np.array(mrc_file.data, dtype=np.float32).flatten(order='C')


        if abs(mrc.widthx - mrc.widthy) > 1e-6 or abs(mrc.widthx - mrc.widthz) > 1e-6 or abs(mrc.widthy - mrc.widthz) > 1e-6:
            print(f"#ERROR: grid sizes are different {mrc.widthx} {mrc.widthy} {mrc.widthz}")
            print("PLEASE USE CUBIC MRC MAP DATA")
            return True

        if mrc.ncstart != 0 or mrc.nrstart != 0 or mrc.nsstart != 0:
            mrc.orgxyz[0] += mrc.ncstart * mrc.widthx
            mrc.orgxyz[1] += mrc.nrstart * mrc.widthy
            mrc.orgxyz[2] += mrc.nsstart * mrc.widthz

    return False



def SetUpVoxSize(m: MRC, M: MRC, t: float, ssize: float, cmd: CMD = None):
    shape = (m.xdim, m.ydim, m.zdim)
    # Step1: 阈值
    if t < 0:
        m.dens -= t
        t = 0.0
    m.dens[m.dens < t] = 0.0

    # Step2: 计算中心
    cent = np.array([m.xdim * 0.5, m.ydim * 0.5, m.zdim * 0.5])

    # Step3: 计算距离中心的最大距离
    x, y, z = np.meshgrid(np.arange(m.xdim), np.arange(m.ydim), np.arange(m.zdim), indexing='ij')
    d2 = (x - cent[0])**2 + (y - cent[1])**2 + (z - cent[2])**2
    dens3d = m.dens.reshape(shape)
    mask = dens3d > 0
    if np.any(mask):
        dmax = np.max(d2[mask])
    else:
        dmax = 0

    # 准备输出内容
    output_lines = []

    # Step4: 新采样中心和步长
    M.cent = cent * m.widthx + m.orgxyz
    M.widthx = ssize

    output_lines.append(f"{M.widthx:.6f}\n")

    m.dmax = np.sqrt(dmax) * m.widthx
    tmp_size = int(2 * np.sqrt(dmax) * m.widthx / M.widthx)

    # Step5: 选择新的网格大小
    a = 2
    while a <= tmp_size:
        a *= 2
    b = 3
    while b <= tmp_size:
        b *= 2
    if b < a:
        a = b
    b = 9
    while b <= tmp_size:
        b *= 2
    if b < a:
        a = b

    M.xdim = M.ydim = M.zdim = a
    M.orgxyz = M.cent - 0.5 * a * M.widthx

    output_lines.extend([
        f"{M.xdim} {M.ydim} {M.zdim}\n",
        f"{M.cent[0]:.6f} {M.cent[1]:.6f} {M.cent[2]:.6f}\n",
        f"{M.orgxyz[0]:.6f} {M.orgxyz[1]:.6f} {M.orgxyz[2]:.6f}\n"
    ])

    # 输出到文件或控制台
    if cmd and cmd.out_file:
        with open(cmd.out_file, 'w') as f:  # 使用追加模式
            f.writelines(output_lines)
    else:
        print(''.join(output_lines), end='')


def fastVEC(m: MRC, M: MRC, cmd: CMD):
    Ndata = M.xdim * M.ydim * M.zdim
    M.vec = np.zeros((Ndata, 3), dtype=np.float64)
    M.dens = np.zeros(Ndata, dtype=np.float32)
    M.xyz = np.zeros((Ndata, 3), dtype=int)

    dreso = cmd.dreso
    gstep = m.widthx
    xydim = m.xdim * m.ydim

    dsum = 0.0
    Nact = 0

    for x in range(M.xdim):
        for y in range(M.ydim):
            for z in range(M.zdim):
                ind = M.xdim * M.ydim * z + M.xdim * y + x
                #pos:当前新采样网格点在原始密度图（m）中的浮点体素索引坐标 pos=（xi*新的采样间隔+新原点-原始原点）/原始的采样间隔
                pos = np.array([
                    (x * M.widthx + M.orgxyz[0] - m.orgxyz[0]) / m.widthx,
                    (y * M.widthx + M.orgxyz[1] - m.orgxyz[1]) / m.widthx,
                    (z * M.widthx + M.orgxyz[2] - m.orgxyz[2]) / m.widthx,
                ])
                #做边界检查，避免越界访问原始密度图。
                if np.any(pos < 0) or pos[0] >= m.xdim or pos[1] >= m.ydim or pos[2] >= m.zdim:
                    M.dens[ind] = 0
                    M.vec[ind, :] = 0
                    continue
                #将pos四舍五入到整数，并计算其在原始密度图中的体素索引ind0
                ind0 = int(m.xdim * m.ydim * int(pos[2]) + m.xdim * int(pos[1]) + int(pos[0]))
                if m.dens[ind0] == 0:
                    M.dens[ind] = 0
                    M.vec[ind, :] = 0
                    continue
                #计算高斯核的参数fs和fsiv，dreso是高斯核带宽（单位是A）
                fs = (dreso / gstep) * 0.5
                fs = fs * fs
                fsiv = 1.0 / fs
                fmaxd = (dreso / gstep) * 2.0

                stp = np.maximum(np.floor(pos - fmaxd), 0).astype(int)
                endp = np.minimum(np.ceil(pos + fmaxd + 1), [m.xdim, m.ydim, m.zdim]).astype(int)

                dtotal = 0.0 #统计加权密度值总和
                pos2 = np.zeros(3, dtype=np.float64) #原始体素的加权坐标总和  坐标*高斯加权*密度值

                #计算采样范围内
                for xp in range(stp[0], endp[0]):
                    rx = (xp - pos[0])**2
                    for yp in range(stp[1], endp[1]):
                        ry = (yp - pos[1])**2
                        for zp in range(stp[2], endp[2]):
                            rz = (zp - pos[2])**2
                            d2 = rx + ry + rz
                            ind2 = xydim * zp + m.xdim * yp + xp
                            v = np.exp(-1.5 * d2 * fsiv) * m.dens[ind2] #高斯加权的密度值
                            dtotal += v
                            pos2[0] += v * xp
                            pos2[1] += v * yp
                            pos2[2] += v * zp

                M.dens[ind] = dtotal
                if dtotal == 0.0:
                    M.vec[ind, :] = 0
                    continue

                rd = 1.0 / dtotal
                pos2 *= rd #归一化 pos2 / dtotal,得到加权重心坐标，可以反映局部形变、配准方向等
                tmpcd = pos2 - pos  #这个矢量是从采样点出发，指向局部加权重心，反映了局部密度的主方向。
                dvec = np.linalg.norm(tmpcd) #计算这个偏移的欧氏距离,即矢量模长
                if dvec == 0.0:
                    dvec = 1.0
                rdvec = 1.0 / dvec
                M.vec[ind, :] = tmpcd * rdvec
                M.xyz[ind, :] = [x, y, z]

                dsum += dtotal
                Nact += 1
                
            
    M.ave = dsum / Nact if Nact > 0 else 0.0

    dsum2 = 0.0
    dsum_sq = 0.0
    for i in range(Ndata):
        if M.dens[i] > 0:
            dsum_sq += (M.dens[i]) * (M.dens[i])
            dsum2 += (M.dens[i] - M.ave) * (M.dens[i] - M.ave)
    M.std_norm_ave = np.sqrt(dsum2)
    M.std = np.sqrt(dsum_sq)
    print(f"{M.ave:.6f} {M.std:.6f} {M.std_norm_ave:.6f}")

    for i in range(Ndata):
        if M.xyz[i, 0] != 0:
            print(f"{i} {M.xyz[i, 0]} {M.xyz[i, 1]} {M.xyz[i, 2]}")
            print(f"{M.vec[i, 0]:.6f} {M.vec[i, 1]:.6f} {M.vec[i, 2]:.6f} {M.dens[i]:.6f}")



@njit(parallel=True)
def fastVEC_numba(
    M_xdim, M_ydim, M_zdim, M_widthx, M_orgxyz,
    m_xdim, m_ydim, m_zdim, m_widthx, m_orgxyz, m_dens,
    dreso,
    vec, dens, xyz
):
    Ndata = M_xdim * M_ydim * M_zdim
    gstep = m_widthx
    xydim = m_xdim * m_ydim

    for x in prange(M_xdim):
        for y in range(M_ydim):
            for z in range(M_zdim):
                ind = M_xdim * M_ydim * z + M_xdim * y + x
                pos = np.empty(3, dtype=np.float64)
                pos[0] = (x * M_widthx + M_orgxyz[0] - m_orgxyz[0]) / m_widthx
                pos[1] = (y * M_widthx + M_orgxyz[1] - m_orgxyz[1]) / m_widthx
                pos[2] = (z * M_widthx + M_orgxyz[2] - m_orgxyz[2]) / m_widthx

                if pos[0] < 0 or pos[1] < 0 or pos[2] < 0 or pos[0] >= m_xdim or pos[1] >= m_ydim or pos[2] >= m_zdim:
                    dens[ind] = 0
                    vec[ind, :] = 0
                    continue

                ind0 = int(m_xdim * m_ydim * int(pos[2]) + m_xdim * int(pos[1]) + int(pos[0]))
                if m_dens[ind0] == 0:
                    dens[ind] = 0
                    vec[ind, :] = 0
                    continue

                fs = (dreso / gstep) * 0.5
                fs = fs * fs
                fsiv = 1.0 / fs
                fmaxd = (dreso / gstep) * 2.0

                stp0 = int(max(np.floor(pos[0] - fmaxd), 0))
                stp1 = int(max(np.floor(pos[1] - fmaxd), 0))
                stp2 = int(max(np.floor(pos[2] - fmaxd), 0))
                end0 = int(min(np.ceil(pos[0] + fmaxd + 1), m_xdim))
                end1 = int(min(np.ceil(pos[1] + fmaxd + 1), m_ydim))
                end2 = int(min(np.ceil(pos[2] + fmaxd + 1), m_zdim))

                dtotal = 0.0
                pos2 = np.zeros(3, dtype=np.float64)
                for xp in range(stp0, end0):
                    rx = (xp - pos[0]) ** 2
                    for yp in range(stp1, end1):
                        ry = (yp - pos[1]) ** 2
                        for zp in range(stp2, end2):
                            rz = (zp - pos[2]) ** 2
                            d2 = rx + ry + rz
                            ind2 = xydim * zp + m_xdim * yp + xp
                            v = np.exp(-1.5 * d2 * fsiv) * m_dens[ind2]
                            dtotal += v
                            pos2[0] += v * xp
                            pos2[1] += v * yp
                            pos2[2] += v * zp

                dens[ind] = dtotal
                if dtotal == 0.0:
                    vec[ind, :] = 0
                    continue

                rd = 1.0 / dtotal
                pos2[0] *= rd
                pos2[1] *= rd
                pos2[2] *= rd
                tmpcd0 = pos2[0] - pos[0]
                tmpcd1 = pos2[1] - pos[1]
                tmpcd2 = pos2[2] - pos[2]
                dvec = np.sqrt(tmpcd0 * tmpcd0 + tmpcd1 * tmpcd1 + tmpcd2 * tmpcd2)
                if dvec == 0.0:
                    dvec = 1.0
                rdvec = 1.0 / dvec
                vec[ind, 0] = tmpcd0 * rdvec
                vec[ind, 1] = tmpcd1 * rdvec
                vec[ind, 2] = tmpcd2 * rdvec
                xyz[ind, 0] = x
                xyz[ind, 1] = y
                xyz[ind, 2] = z


def fastVEC_numba_wrapper(m: MRC, M: MRC, cmd: CMD):
    """
    封装fastVEC_numba函数，提供与fastVEC相同的接口和功能
    
    Args:
        m: 输入MRC对象
        M: 输出MRC对象
        cmd: 命令行参数对象
    """
    Ndata = M.xdim * M.ydim * M.zdim
    M.vec = np.zeros((Ndata, 3), dtype=np.float64)
    M.dens = np.zeros(Ndata, dtype=np.float32)
    M.xyz = np.zeros((Ndata, 3), dtype=np.int32)

    # 调用numba加速的核心函数
    fastVEC_numba(
        M.xdim, M.ydim, M.zdim, M.widthx, M.orgxyz.astype(np.float64),
        m.xdim, m.ydim, m.zdim, m.widthx, m.orgxyz.astype(np.float64), m.dens.astype(np.float32),
        cmd.dreso,
        M.vec, M.dens, M.xyz
    )

    # 计算统计值
    valid = M.dens > 0
    if valid.any():
        M.ave = M.dens[valid].mean()
        M.std = np.sqrt((M.dens[valid] ** 2).sum())
        M.std_norm_ave = np.sqrt(((M.dens[valid] - M.ave) ** 2).sum())
    else:
        M.ave = 0.0
        M.std = 0.0
        M.std_norm_ave = 0.0


    # 找到所有非零点的索引
    nonzero_indices = np.where(M.xyz[:, 0] != 0)[0]
    
    # 获取非零点的数据
    xyz_nonzero = M.xyz[nonzero_indices]
    vec_nonzero = M.vec[nonzero_indices]
    dens_nonzero = M.dens[nonzero_indices]
    
    # 按照x,y,z坐标排序（从小到大）
    sort_idx = np.lexsort((xyz_nonzero[:, 2], xyz_nonzero[:, 1], xyz_nonzero[:, 0]))
    
    # 准备输出内容
    output_lines = []
    output_lines.append(f"{M.ave:.6f} {M.std:.6f} {M.std_norm_ave:.6f}\n")

    # 按排序顺序输出非零点信息
    for idx in sort_idx:
        i = nonzero_indices[idx]
        output_lines.append(f"{i} {xyz_nonzero[idx, 0]} {xyz_nonzero[idx, 1]} {xyz_nonzero[idx, 2]}\n")
        output_lines.append(f"{vec_nonzero[idx, 0]:.6f} {vec_nonzero[idx, 1]:.6f} {vec_nonzero[idx, 2]:.6f} {dens_nonzero[idx]:.6f}\n")

    # 如果指定了输出文件，写入文件，否则打印到控制台
    if cmd.out_file:
        with open(cmd.out_file, 'a') as f:
            f.writelines(output_lines)
    else:
        print(''.join(output_lines), end='')


def ShowVec(M: MRC):
    for x in range(M.xdim):
        for y in range(M.ydim):
            for z in range(M.zdim):
                ind = M.xdim * M.ydim * z + M.xdim * y + x
                if M.dens[ind] == 0.0 and ind != 0:
                    continue
                tmp = np.array([x * M.widthx + M.orgxyz[0],
                                y * M.widthx + M.orgxyz[1],
                                z * M.widthx + M.orgxyz[2]])
                print(f"H       {tmp[0]:.6f}        {tmp[1]:.6f}        {tmp[2]:.6f}")
                tmp = np.array([(x + M.vec[ind, 0]) * M.widthx + M.orgxyz[0],
                                (y + M.vec[ind, 1]) * M.widthx + M.orgxyz[1],
                                (z + M.vec[ind, 2]) * M.widthx + M.orgxyz[2]])
                print(f"H       {tmp[0]:.6f}        {tmp[1]:.6f}        {tmp[2]:.6f}")

def Sample(file, threshold, voxel_size, dreso, output_file, mode=False):
    cmd = CMD()
    cmd.file = file
    cmd.dreso = dreso
    cmd.th1 = threshold
    cmd.ssize = voxel_size
    cmd.Nthr = 1
    cmd.Mode = mode
    cmd.out_file = output_file

    mrc = MRC()
    mrcN1 = MRC()
    if readmrc(mrc, cmd.file):
        return
    SetUpVoxSize(mrc, mrcN1, cmd.th1, cmd.ssize, cmd)
    fastVEC_numba_wrapper(mrc, mrcN1, cmd)

    if cmd.Mode == 1:
        ShowVec(mrcN1)
    
    

def main():
    args = parse_args()
    cmd = CMD()
    cmd.file = args.file
    cmd.th1 = args.th1
    cmd.dreso = args.dreso
    cmd.ssize = args.ssize
    cmd.Nthr = args.Nthr
    cmd.Mode = 1 if args.Mode else 0
    cmd.out_file = args.out_file

    t1 = time.time()
    mrc = MRC()
    mrcN1 = MRC()

    if readmrc(mrc, cmd.file):
        return
    SetUpVoxSize(mrc, mrcN1, cmd.th1, cmd.ssize, cmd)
    
    # 使用封装后的函数
    fastVEC_numba_wrapper(mrc, mrcN1, cmd)
    #fastVEC(mrc, mrcN1, cmd)


    if cmd.Mode == 1:
        ShowVec(mrcN1)
    t4 = time.time()
    print(f"#FINISHED TOTAL TIME= {t4 - t1:.6f}")

if __name__ == "__main__":
    main()
