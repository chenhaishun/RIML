import numpy as np
from scipy.spatial.distance import cdist
from util import mv_dataset, read_mymat, build_ad_dataset, process_data, get_validation_set, \
    mv_tabular_collate
from sklearn.linear_model import LogisticRegression
import torch
from Sinkhorn_distance import SinkhornDistance, SinkhornDistance_one_to_multi,SinkhornDistance_another_one_to_multi,SinkhornDistance_given_cost
import cupy as cp

use_gpu = torch.cuda.is_available()
device = 'cuda:0' if (torch.cuda.is_available()) else 'cpu:0'

d_cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)

eplison=0.01
sinkhorn = SinkhornDistance(eps=eplison, max_iter=200, reduction=None).to(device)  

def tukeys_transform(data, beta=0.5): 
    return np.power(data, beta)

reg_param = 1e-3

def distribution_calibration_dan(prototype, probabi, base_means, base_cov, n_lsamples,alpha=0.01,lambd=0.3,k=10):

    index = np.argsort(-probabi.cpu().numpy())
    dim=base_means[0].shape[0]
    calibrated_mean=0
    calibrated_cov=0

    proab_reshape=np.repeat(n_lsamples*probabi.cpu().numpy(),dim,axis=0).reshape(len(base_means),dim)
    calibrated_mean= (1-lambd)*np.sum(proab_reshape *np.concatenate([base_means[:]]),axis=0)+lambd*prototype
    #
    proab_reshape_conv=np.repeat(n_lsamples*probabi.cpu().numpy(),dim*dim,axis=0).reshape(len(base_means),dim,dim)
    calibrated_cov=np.sum(proab_reshape_conv*np.concatenate([base_cov[:]]),axis=0)+alpha
    return calibrated_mean, calibrated_cov


def get_ot_samples(x, y, sn, train_index, test_index, n_sample, k, if_test=False):

    view_num = len(x)
    data_num = x[0].shape[0]
    print("calculate distance")
    dist_all_set = [cdist(x[i], x[i], 'euclidean') for i in range(view_num)]

    dismiss_view_index = [[np.array([]) for __ in range(view_num)] for _ in range(view_num)]  
    for i in range(view_num-1):
        for j in range(i+1, view_num):
            sn_temp = sn[:, [i, j]]
            sn_temp_sum = np.sum(sn_temp, axis=1)
            sn_temp_sum[test_index] = 0  # Mask the test set sample
            dismiss_view_index[i][j] = dismiss_view_index[j][i] = np.where(sn_temp_sum == 2)[0] 
    
    sn_train = sn[train_index]
    x_train = [x[v][train_index] for v in range(view_num)]
    y_train = y[train_index]

    class_num = np.max(y) + 1
    means, covs, num, label, dataset, shape_all = dict(), dict(), dict(), dict(), dict(), dict()
    # s_data, s_label = dict(), dict()
    support_data, support_label = dict(), dict()
    base_means, base_covs, base_data, base_label = [], [], [], []

    x_train_dissmiss_index = np.where(np.sum(sn_train, axis=1) == view_num)[0]
    x_complete = [x_train[_][x_train_dissmiss_index] for _ in range(view_num)]
    y_complete = y_train[x_train_dissmiss_index]
    sn_train_complete = sn_train[x_train_dissmiss_index]

    x_train_miss_index = np.where(np.sum(sn_train, axis=1) < view_num)[0]
    x_incomplete = [np.repeat(x_train[_][x_train_miss_index], n_sample, axis=0) for _ in range(view_num)]
    y_incomplete = np.repeat(y_train[x_train_miss_index], n_sample, axis=0)
    sn_train_incomplete = np.repeat(sn_train[x_train_miss_index], n_sample, axis=0)
    
    if if_test is not True:

        print("Fill the missing views in the training set")

        for v in range(view_num):
            present_index = np.where(sn_train[:, v] == 1)[0]  
            means_v, covs_v, data_v, label_v = [], [], [], []
            support_data_v, support_label_v = [], []
            shape_v=[]
            for c in range(class_num):
                present_index_class = np.where(y_train[present_index] == c)[0] 
                if len(present_index_class) > 50:
                    mean_v = np.mean(x_train[v][present_index_class], axis=0)
                    cov_v = np.cov(x_train[v][present_index_class], rowvar=0)
                    means_v.append(mean_v)
                    covs_v.append(cov_v)
                    data_v.append(x_train[v][present_index_class])
                    label_v.extend([c] * len(present_index_class))
                    shape_v.append(x_train[v][present_index_class].shape[0])
                else:
                    means_v.append(None)
                    covs_v.append(None)
                    support_data_v.append(x_train[v][present_index_class]) 
                    support_label_v.append([c] * len(present_index_class))

            means[v], covs[v], shape_all[v] = means_v, covs_v, shape_v
            support_data[v] = np.concatenate(support_data_v, axis=0)
            support_label[v] = np.concatenate(support_label_v, axis=0)
            dataset[v] = np.concatenate(data_v, axis=0) 
        
            base_means.append([mean for mean in means_v if mean is not None])
            base_covs.append([cov for cov in covs_v if cov is not None])
            base_data.append([data for data in data_v if data is not None])
            base_label.append([label for label in label_v if label is not None])

        # Tukey's transform 
        for v in range(view_num):
            support_data[v] = tukeys_transform(support_data[v])

        for v in range(view_num):
            classifier_base = LogisticRegression(max_iter=1000).fit(X=dataset[v], y=base_label[v])
            feature_all_reshape=np.zeros(shape=[len(base_means[v]), max(shape_all[v]), dataset[v].shape[1]]) 
            sample_probabi=torch.zeros([len(base_means[v]),max(shape_all[v])])

            for ii in range(len(base_data[v])): 
                if shape_all[v][ii]==max(shape_all[v]):
                    feature_all_reshape[ii,:,:] = base_data[v][ii]
                else:
                    pad = max(shape_all[v])-shape_all[v][ii]
                    Tmp = base_data[v][ii]
                    if pad < shape_all[v][ii]:  
                        Tmp1 =np.concatenate([Tmp, Tmp[0:pad]])
                    else:
                        addition_pad = pad - shape_all[v][ii]
                        selected_samples = Tmp[0:shape_all[v][ii]]
                        Tmp_a = Tmp[np.random.choice(Tmp.shape[0], addition_pad, replace=True)] # False 
                        Tmp1 =np.concatenate([Tmp, Tmp_a, selected_samples],axis=0)
                    feature_all_reshape[ii,:,:] = Tmp1
                predict_prob = classifier_base.predict_proba(feature_all_reshape[ii,:,:])
                sample_probabi[ii,:] = torch.nn.functional.softmax(torch.from_numpy(predict_prob[:, ii]/0.3), dim=0) 

            feature_torch = torch.from_numpy(feature_all_reshape).to(device)
            base_torch=torch.from_numpy(np.concatenate([base_means[v][:]])).to(device)
            support_torch = torch.from_numpy(support_data[v]).to(device)
            support_each = torch.from_numpy(support_data[v]).to(device).unsqueeze(0)

            # Sinkhorn_distance in each view
            cost, Pi, C = sinkhorn(base_torch, support_torch)
            print("a view completed")
            n_lsamples = support_data[v].shape[0]

            for label in np.unique(support_label[v]):
                label_indices = np.where(support_label[v] == label)[0]
                means_list = []
                covs_list = []
                for num in label_indices:
                    mean, cov = distribution_calibration_dan(support_data[v][num], Pi[:, num], base_means[v], base_covs[v], n_lsamples, alpha=0.21, lambd=0.7, k=len(base_means[v]))
                    means_list.append(mean)
                    covs_list.append(cov)

                means[v][label]  = np.mean(np.stack(means_list), axis=0)
                covs[v][label] = np.mean(np.stack(covs_list), axis=0)

        index = 0
        for i in x_train_miss_index:
            y_i = y_train[i][0]
            miss_view_index = np.nonzero(sn_train[i] == 0)[0]
            for v in miss_view_index:
                rng = np.random.default_rng()
                cov = covs[v][y_i] + np.eye(len(covs[v][y_i])) * reg_param
                L = np.linalg.cholesky(cov)
                samples_v = rng.normal(size=(n_sample, len(cov))) @ L.T + means[v][y_i]
                x_incomplete[v][index*n_sample:(index+1)*n_sample] = samples_v
            index+=1

    x_train = [np.concatenate((x_complete[_], x_incomplete[_]), axis=0) for _ in range(view_num)]
    y_train = np.concatenate((y_complete, y_incomplete), axis=0)
    Sn_train = np.concatenate((sn_train_complete, sn_train_incomplete), axis=0)

    print("Fill the missing views in the test set")
    sn_test = sn[test_index]
    x_test_dissmiss_index = np.where(np.sum(sn_test, axis=1) == view_num)[0]
    x_test_miss_index = np.where(np.sum(sn_test, axis=1) < view_num)[0]
    sn_test_complete = np.repeat(sn_test[x_test_dissmiss_index], n_sample, axis=0)
    sn_test_incomplete = np.repeat(sn_test[x_test_miss_index], n_sample, axis=0)
    Sn_test = np.concatenate((sn_test_complete, sn_test_incomplete), axis=0)

    # Tukey's transform 
    x_test = [x[v][test_index] for v in range(view_num)]
    for v in range(view_num):
        x_test[v] = tukeys_transform(x_test[v])

    # Impute missing views with all of the multiple sampling points
    x_test = [np.repeat(x[_][test_index][x_test_dissmiss_index], n_sample, axis=0) for _ in range(view_num)]
    y_test = np.repeat(y[test_index][x_test_dissmiss_index], n_sample, axis=0)

    for i in test_index.flat:
        x_i = [np.repeat(np.expand_dims(x[_][i], axis=0), n_sample, axis=0) for _ in range(view_num)]
        y_i = np.repeat(np.expand_dims(y[i], axis=0), n_sample, axis=0)

        sn_temp = sn[i]
        x_miss_view_index = np.nonzero(sn_temp == 0)[0]
        x_dismiss_view_index = np.nonzero(sn_temp)[0]

        if x_miss_view_index.shape[0] != 0:
            for j in x_miss_view_index.flat:
                neighbors_index_temp = np.array([], dtype=np.int_)
                for jj in x_dismiss_view_index.flat:
                    dismiss_view_index_temp = dismiss_view_index[j][jj]
                    dist_temp = np.full(data_num, np.inf)
                    dist_temp[dismiss_view_index_temp] = dist_all_set[jj][i, dismiss_view_index_temp]
                    nearest_index_temp = np.argpartition(dist_temp, k)[:k]
                    neighbors_index_temp = np.unique(np.concatenate((neighbors_index_temp, nearest_index_temp)))

                x_neighbors_temp = x[j][neighbors_index_temp]
                mean = np.mean(x_neighbors_temp, axis=0)
                cov = np.cov(x_neighbors_temp, rowvar=0)
                rng = np.random.default_rng()
                cov = cov + np.eye(len(cov)) * reg_param
                L = np.linalg.cholesky(cov)
                x_samples_temp = rng.normal(size=(n_sample, len(cov))) @ L.T + mean
                x_i[j] = x_samples_temp

            x_test = [np.concatenate((x_test[_], x_i[_]), axis=0) for _ in range(view_num)]
            y_test = np.concatenate((y_test, y_i), axis=0)


    return x_train, y_train, x_test, y_test, Sn_train