import numpy as np
from sklearn.cluster import DBSCAN
from scipy.special import expit



def ls2localPC(ranges, valid_cond=None):
    prox_ordr = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4]
    angs = np.array([(i-1)*np.deg2rad(22.5) for i in prox_ordr])+np.deg2rad(180.0)
    sAng = np.sin(angs)
    cAng = np.cos(angs)

    if valid_cond is None:
        valid_inds_bool = ranges.T<=4.999  # np.logical_and(ranges<=4999,ranges>1)
    else:
        valid_inds_bool = valid_cond(ranges.T)

    valid_inds_bool_r = np.ravel(valid_inds_bool)  # full, logical, raveled
    valid_inds, _ = np.meshgrid(np.arange(ranges.shape[0]), np.ones((16,1)))  # time indexes, 
    valid_inds = np.ravel(valid_inds)[valid_inds_bool_r]  # time indexes, raveled, partial

    pc_x = np.diag(cAng) @ ranges.T
    pc_y = np.diag(sAng) @ ranges.T

    return pc_x, pc_y, valid_inds_bool, valid_inds
    

def ls2pc(x, y, t, ranges, valid_cond=None):
    pc_x, pc_y, valid_inds_bool, valid_inds = ls2localPC(ranges, valid_cond=valid_cond)
    cTheta = np.outer(np.ones(16,),np.cos(t))
    sTheta = np.outer(np.ones(16,),np.sin(t))
    pc_x_r = np.multiply(pc_x,cTheta)-np.multiply(pc_y,sTheta) + np.outer(np.ones((16,)), x)
    pc_y_r = np.multiply(pc_x,sTheta)+np.multiply(pc_y,cTheta) + np.outer(np.ones((16,)), y)

    return pc_x_r, pc_y_r, valid_inds_bool, valid_inds


def ls2ft(ranges, th=0.2, lookingUp=False):
    f = lambda x: np.logical_and(x>0.01, x<4.99)
    pc_x_n0, pc_y_n0, _, _ = ls2localPC(ranges, lookingUp=lookingUp, valid_cond=f)
    # pc_x_n0.shape = (16, L)    (  also valid_inds_bool  )
    return calc_lin_char(pc_x_n0, pc_y_n0, ranges, th=th)

def calc_lin_char(pc_x_n2, pc_y_n2, ranges, th=0.2):
    pc_x_n1 = np.roll(pc_x_n2, -1, axis=0)
    pc_y_n1 = np.roll(pc_y_n2, -1, axis=0)
    pc_x_n3 = np.roll(pc_x_n2, 1, axis=0)
    pc_y_n3 = np.roll(pc_y_n2, 1, axis=0)

    num = (pc_x_n3-pc_x_n1)*(pc_x_n2-pc_x_n3)+(pc_y_n3-pc_y_n1)*(pc_y_n2-pc_y_n3)
    den = (pc_y_n3-pc_y_n1)**2+(pc_x_n3-pc_x_n1)**2

    alpha = - num / den
    # print(f'alpha={alpha}')
    L_x = (pc_x_n2-pc_x_n3) + alpha * (pc_x_n3-pc_x_n1)
    L_y = (pc_y_n2-pc_y_n3) + alpha * (pc_y_n3-pc_y_n1)

    d = (L_x**2 + L_y**2)**0.5
    # print(f'd={d}, ds shape ={d.shape}')

    flatWall_indicator = np.zeros_like(ranges.T)
    rng_non_zero = np.clip(ranges.T, a_min=0.1, a_max=np.Inf)
    flatWall_indicator[np.where(np.logical_and(np.logical_and(d/(rng_non_zero) < th, 0.01 < ranges.T), ranges.T< 4))] = 1.0
    fw_count = np.sum(flatWall_indicator-np.roll(flatWall_indicator, -1, axis=0)>0, axis=0)
    return np.sum(flatWall_indicator, axis=0), fw_count





# A feature has geometrical information and angular data
from enum import Enum
class featType(Enum):
    linear = 1
    narrow = 2
    none = 3
    multi_cluster = 4

def clusterAnalysis(pc, eps=0.5, min_samples=1):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(pc)
    labels = db.labels_
    classes, cnt = np.unique(labels, return_counts=True)
    classes = sorted(classes, key=lambda x: cnt[x], reverse=True)
    cnt = sorted(cnt, reverse=True)
    if min_samples>1:  # remove outliers from stats
        cnt = [c for c, l in zip(cnt, classes) if l!=-1]
        classes = [l for l in classes if l!=-1]

    return labels, classes, cnt



def getFeaturesRanges(ranges, t, lookingUp=False, valid_cond=None):
    if lookingUp:
        prox_ordr = [5, 4, 3, 2, 1, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]
        angs = np.array([(i-1)*np.deg2rad(22.5) for i in prox_ordr])+np.deg2rad(180.0)*0
    else:
        prox_ordr = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2, 3, 4]
        angs = np.array([(i-1)*np.deg2rad(22.5) for i in prox_ordr])+np.deg2rad(180.0)

    if valid_cond is None:
        valid_inds_bool = np.logical_and(ranges.T>0.01, ranges.T<=4.999)  # np.logical_and(ranges<=4999,ranges>1)
    else:
        valid_inds_bool = valid_cond(ranges.T)

    ranges_s = np.roll(ranges,shift=8)
    valid_inds_bool_s = np.roll(valid_inds_bool,shift=8)

    feat_data, feat_type = [], []
    rr = ranges[:8]+5.0*(1.0-valid_inds_bool[:8])+ranges_s[:8]+5.0*(1.0-valid_inds_bool_s[:8])
    nrw_pass = 1.0*(rr < 0.6)+1.0*(rr < 0.8)+1.0*(rr < 1.2)
    ind_min = np.argmin(rr)
    if nrw_pass[ind_min]>0:
        feat_type.append(featType.narrow)
        feat_data.append([nrw_pass[ind_min], np.mod(t+angs[ind_min]+np.pi/2,np.pi)-np.pi/2])




    # ranges_p1 = np.roll(ranges+10.0*(1-valid_inds_bool),shift=+1)
    # ranges_p2 = np.roll(ranges+10.0*(1-valid_inds_bool),shift=+2)
    # ranges_n1 = np.roll(ranges+10.0*(1-valid_inds_bool),shift=-1)
    # ranges_n2 = np.roll(ranges+10.0*(1-valid_inds_bool),shift=-2)
    # max_p = np.max([ranges_p1, ranges_p2], axis=0)
    # max_n = np.max([ranges_n1, ranges_n2], axis=0)
    # minmax = np.min([max_p, max_n], axis=0)

    # mashkof = minmax[:8]-ranges[:8]-(1-valid_inds_bool[:8])*10
    # inds_mashkof = np.argsort(-mashkof)
    # for i in inds_mashkof:
    #     if mashkof[i]<0.2:
    #         if rr[i]<1.6:
    #             ind_min = i
    #             pass_width_type = 1.0*(rr[ind_min]<0.8)+1.0*(rr[ind_min]<1.3)+1.0*(rr[ind_min]<1.6)
    #             feat_type.append(featType.narrow)
    #             feat_data.append([pass_width_type, np.mod(t+angs[ind_min]+np.pi/2,np.pi)-np.pi/2])
    #             break
    #     else:
    #         break
    
    return feat_data, feat_type

    # rr_ = rr
    # nrw_pass = 1.0*(rr_ < 0.6)+1.0*(rr_ < 0.8)+1.0*(rr_ < 1.2)
    # mashkof*nrw_pass

    # nrwness = expit()
    # inds = np.where(minmax-ranges>0.3 and valid_inds_bool and rr<1.6)

    # ind_min = np.argmin(rr[:8])
    # if ind_min<2:
    #     ind_min +=8
    # elif ind_min>14:
    #     ind_min -=8
    # t1, t2 = False, False
    # f = lambda r, i: (r[i]+0.3<np.min([np.max(r[[i-2, i-1]]),np.max(r[[i+1, i+2]])]))
    # if f(ranges,ind_min):
    #     t1 = True
    # if f(ranges_s,ind_min):
    #     t2 = True
    pass_width_type = 1.0*(rr[ind_min]<0.8)+1.0*(rr[ind_min]<1.3)+1.0*(rr[ind_min]<1.6)
    if ( (t1 or t2) and (pass_width_type>0.0)):
        feat_type.append(featType.narrow)
        feat_data.append([pass_width_type, np.mod(t+angs[ind_min]+np.pi/2,np.pi)-np.pi/2])
    
    # rng_ = np.array([r for r in ranges if r>0.1])
    # d_max_sided_p = np.maximum(ranges_p1,ranges_p2)
    # d_max_sided_n = np.maximum(ranges_n1,ranges_n2)
    # d_min = np.minimum(rng_-d_max_sided_p, rng_-d_max_sided_n)
    # d_min_cnt = np.sum(np.logical_and(d_min>0.2, rng_<2.0))
    # if d_min_cnt>=2:
    #     feat_data.append(d_min_cnt)
    #     feat_type.append(featType.multi_cluster)
    
    return feat_data, feat_type
    

def test_getFeaturesRanges():
    ranges = np.ones(shape=(16,))*3.0
    ranges[1]=1.2
    ranges[0]=1.0
    ranges[-1]=1.2
    ranges[-2]=1.2
    # ranges[2]=1.2
    ranges[8]=1
    ranges[9]=1.1
    ang, feat = getFeaturesRanges(ranges, np.deg2rad(10.0))
    # print(f'feat={feat}, ang={np.rad2deg(ang[0])}')


def getFeatures(pc_i, min_cluster_size=3, eps=0.5):
    if pc_i.shape[0]<3:
        return [], []

    labels, classes, cnt = clusterAnalysis(pc_i, eps=eps)
    num_classes_counter = 0
    linear = 0
    non_linear = 0
    vert = 0
    phi_tmp = []
    feat_data = []
    feat_type = []
    for cls, cnt_ in zip(classes, cnt):
        if cnt_ < min_cluster_size:
            break
        else:
            num_classes_counter += 1
            pc_tmp = pc_i[np.array(labels)==cls,:]
            rb_ = np.dot(np.linalg.pinv(pc_tmp), np.ones(shape=(cnt_,)))
            rho = 1/np.sqrt(rb_[0]**2+rb_[1]**2)
            if rb_[1]>=0:
                phi = np.math.atan2(-rb_[0], rb_[1])
            else:
                phi = np.math.atan2(rb_[0], -rb_[1])
            
            err = np.abs(np.dot(pc_tmp, rb_)*rho-rho)
            ts = np.dot(pc_tmp, np.array([-rb_[1], rb_[0]]))*rho

            is_linear = np.max(err)<0.05
            if is_linear:
                linear += 1
                phi_tmp.append(phi)
                feat_data.append([rho, phi])
                feat_type.append(featType.linear)
            if np.max(err)>0.05:
                non_linear += 1
            if is_linear and (np.abs(phi-np.deg2rad(0))<np.deg2rad(10)):
                vert += 1
            
            if False:
                cov = np.cov(pc_tmp.T)
                w, v = np.linalg.eig(cov)
                if w[0]<0.0001*w[1]:
                    ratio_eigval = 1/0.0001
                else:
                    ratio_eigval = w[1]/w[0]

                if ratio_eigval<100:
                    print(f'ratio_eigval={ratio_eigval}')
                    print(f'err = {err}')
                    print('')
    
    labels, classes, cnt = clusterAnalysis(pc_i, eps=0.6)
    for i,c in enumerate(cnt):
        if c<2:
            if i>=3:
                feat_data.append(i-1)
                feat_type.append(featType.multi_cluster)
            break

    return feat_data, feat_type


def get_sigmoid(percent_loc, percent_val):
    assert(percent_val<1.0 and percent_val>0.5)
    xx = np.linspace(-10,10,500)
    yy = expit(xx)
    # print(f'shape(np.where(yy>percent_val))={np.where(yy>percent_val)[0].shape}')
    s = np.where(yy>percent_val)[0][0]
    # print(f's={s}, (xx[s]={xx[s]})')
    sf = xx[s]/percent_loc
    def sig(x):
        return expit(x*sf)
    return sig

myExpit = get_sigmoid(0.3, 0.9)
isFar = lambda x: myExpit(x-1.0)

nrwExpit = get_sigmoid(0.7, 0.9)
nrwScr = lambda x: nrwExpit(0.6-x)

def sceneIsFar(ranges):
    return np.sum(isFar(ranges))

# xx = np.linspace(-10, 15,300)
# sgmd_1 = get_sigmoid(5.0,0.9)
# sgmd_2 = get_sigmoid(2.0,0.9)
# plt.figure()
# plt.plot(xx, sgmd_1(xx-5))
# plt.plot(xx, 1-(sgmd_2(xx-4)+sgmd_2(-4-xx)))
# plt.show()

sgmd_close = get_sigmoid(percent_loc=0.2,percent_val=0.9)


def get_features(ranges, pc_x_r, pc_y_r):
    
    # shift measurements:
    sig_1_mat = np.roll(ranges, -1, axis=1)
    sig_2_mat = np.roll(ranges, 1, axis=1)
    sig_3_mat = np.roll(ranges, -2, axis=1)
    sig_4_mat = np.roll(ranges, 3, axis=1)
    sig_7_mat = np.roll(ranges, 8, axis=1)

    # calculate median 3 and 5:
    rs_med3_mat = np.median(np.concatenate((np.expand_dims(ranges,axis=2), np.expand_dims(sig_1_mat,axis=2), np.expand_dims(sig_2_mat,axis=2)), axis=2), axis=2)

    rs_med5_mat = np.median(np.concatenate((np.expand_dims(ranges,axis=2),
                                        np.expand_dims(sig_1_mat,axis=2),
                                        np.expand_dims(sig_2_mat,axis=2),
                                        np.expand_dims(sig_3_mat,axis=2),
                                        np.expand_dims(sig_4_mat,axis=2)),
                                        axis=2), axis=2)
    
    pc_x_r_matRoll1 = np.roll(pc_x_r, 1, axis=0)
    pc_x_r_matRolln1 = np.roll(pc_x_r, -1, axis=0)
    pc_y_r_matRoll1 = np.roll(pc_y_r, 1, axis=0)
    pc_y_r_matRolln1 = np.roll(pc_y_r, -1, axis=0)

    feature_2_mat =np.sum(0.5+0.5*np.tanh(4*np.multiply(np.abs(pc_x_r_matRoll1*0.5+pc_x_r_matRolln1*0.5-pc_x_r),(ranges.T<2.0))+np.multiply(np.abs(pc_y_r_matRoll1*0.5+pc_y_r_matRolln1*0.5-pc_y_r),(ranges.T<2.0))-0.3), axis=0).T

    # feature_1_mat = np.sum(np.abs(ranges-rs_med3_mat)>1.0, axis=1)
    simple_calc = False
    if simple_calc:
        sig_7 = 0.5*(sig_7_mat+ranges<1.2)+0.5*(sig_7_mat+ranges<0.9)
        feature_7_mat = np.sum(sig_7, axis=1)
    else:
        sig_7 = sgmd_close(1.0-(sig_7_mat+ranges))
        feature_7_mat = np.sum(sig_7, axis=1)

    sig_7 = sgmd_close(1.0-(sig_7_mat+ranges))
    feature_7_mat = np.sum(sig_7, axis=1)

    sig_8 = sgmd_close(2.0-(sig_7_mat+ranges))-sig_7
    feature_8_mat = np.sum(sig_8, axis=1)

    sig_9 = sgmd_close(3.0-(sig_7_mat+ranges))-sig_8
    feature_9_mat = np.sum(sig_9, axis=1)

    sig_10 = sgmd_close(4.0-(sig_7_mat+ranges))-sig_9
    feature_10_mat = np.sum(sig_10, axis=1)

    sig_11 = sgmd_close(5.0-(sig_7_mat+ranges))-sig_10
    feature_11_mat = np.sum(sig_11, axis=1)


    feature_6_mat = np.sum(0.4*(np.abs(ranges-rs_med5_mat)>1.7)+0.3*(np.abs(ranges-rs_med5_mat)>1.2)+0.3*(np.abs(ranges-rs_med5_mat)>0.5), axis=1)
    feature_3_mat = np.sum(np.logical_and(np.array(ranges<2.0,dtype=float), np.array(np.abs(0.5*sig_1_mat+0.5*sig_2_mat-ranges)<0.3,dtype=float)), axis=1)
    feature_1_mat = np.sum(np.logical_and(np.array(ranges<2.0,dtype=float), np.array(np.abs(0.25*sig_3_mat+0.25*sig_4_mat+0.25*sig_1_mat+0.25*sig_2_mat-ranges)<0.3,dtype=float)), axis=1)

    if True:
        feature_4_mat = np.array(rs_med3_mat>3.0, dtype=float)
        feature_4_mat = np.sum(feature_4_mat-np.roll(feature_4_mat, 1, axis=1)>0, axis=1)
    else:
        sgmd_close4 = get_sigmoid(percent_loc=0.7,percent_val=0.9)
        fm4_ = sgmd_close4(rs_med3_mat-3.0)
        feature_4_mat = np.sum(sgmd_close4(fm4_-np.roll(fm4_, 1, axis=1)-0.98), axis=1)


    feature_5_mat = np.array(rs_med5_mat>3.0, dtype=float)
    feature_5_mat = np.sum(feature_5_mat-np.roll(feature_5_mat, 1, axis=1)>0, axis=1)
    d1 = feature_4_mat*0.5+0.5*feature_5_mat
    d2 = feature_2_mat
    d3 = feature_1_mat

    fw, fw_count = calc_lin_char(pc_x_r, pc_y_r, ranges, th=0.2)
    feature_12_mat, feature_13_mat = fw, fw_count

    discriptor = np.concatenate((   np.expand_dims(d1,axis=1),
                                    np.expand_dims(d2,axis=1), 
                                    np.expand_dims(d3,axis=1), 
                                    np.expand_dims(feature_3_mat,axis=1),
                                    np.expand_dims(feature_6_mat,axis=1),
                                    np.expand_dims(feature_7_mat,axis=1),
                                    np.expand_dims(feature_8_mat,axis=1),
                                    np.expand_dims(feature_9_mat,axis=1),
                                    np.expand_dims(feature_10_mat,axis=1),
                                    np.expand_dims(feature_11_mat,axis=1),
                                    np.expand_dims(feature_12_mat,axis=1),
                                    np.expand_dims(feature_13_mat,axis=1)), axis=1)

    return (feature_1_mat, feature_2_mat, feature_3_mat, feature_4_mat, feature_5_mat, feature_6_mat), discriptor

def get_features_from_ranges(ranges):
    pc_x, pc_y, _, _ = ls2localPC(ranges)
    _, descriptor = get_features(ranges, pc_x, pc_y)
    return descriptor



if __name__=="__main__":
    from matplotlib import pyplot as plt
    import matplotlib.cm as cm
    import numpy as np
    import pandas as pd
    if False:
        '''
        real life test:
        '''

        file_list = [['rec11.csv', False],  #NHSHOLIM
        ['rec11.csv', False]]  # NGB

        for file_name, lookingUp in file_list:
            table = pd.read_table(file_name, delimiter=',')
            ranges = table[[f'mr18.m{i}' for i in range(16)]].to_numpy(dtype=np.float)*0.001
            x = table['stateEstimate.x'].to_numpy()
            y = table['stateEstimate.y'].to_numpy()
            t = np.unwrap(np.deg2rad(table['stateEstimate.yaw'].to_numpy()))
            fw, fw_count = ls2ft(ranges, th=0.2)
            pc_x_r, pc_y_r, valid_inds_bool, valid_inds = ls2pc(x, y, t, ranges, lookingUp=False, valid_cond=None)
            plt.figure()
            plt.plot(pc_x_r[valid_inds_bool][::11], pc_y_r[valid_inds_bool][::11], 'm.', markersize=1)
            # plt.scatter(x, y, c=fw, marker='o')
            plt.scatter(x, y, c=fw_count, cmap='viridis')
            plt.colorbar()
            plt.show()
    elif False:
        '''
        simple test:
        '''
        pc = np.array([[1,2,3,2], [1,2,3,4], [1,1,1,1]]).T
        pc_x_r, pc_y_r = pc[:,[0]], pc[:,[1]]
        print(f'pc_x_r.shape={pc_x_r.shape}')
        ranges = pc[:,[2]].T
        fwi, fw_cnt = calc_lin_char(pc_x_r, pc_y_r, ranges, th=0.2)
        print(f'fwi={fwi}')
        print(f'fw_cnt={fw_cnt}')
        plt.figure()
        plt.scatter(pc_x_r, pc_y_r)
        plt.show()
    elif True:
        test_getFeaturesRanges()