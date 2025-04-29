import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
import random
import random

# 设置随机种子
random.seed(42)
  

# # TODO: Implement get_dynamic_sn
# def get_dynamic_sn(view_num, alldata_len, missing_rate):
#     '''where the missing rate is related to the number of samples, 
#     the more the number of samples, the greater the missing rate, and vice versa'''
def calculate_missing_rates(category_counts, max_missrate):
    """
    根据幂律分布计算缺失率。

    :param category_counts: 类别及其样本数量的字典
    :param max_missrate: 最大缺失率
    :param min_samples: 最少样本数量
    :param max_samples: 最多样本数量
    :return: 每个类别的缺失率字典
    """
    # 计算样本数量的最大值和最小值
    max_count = max(category_counts.values())
    min_count = min(category_counts.values())
    
    # 如果所有类别的样本数量相同，则直接返回最大缺失率
    if max_count == min_count:
        return {category: max_missrate for category in category_counts}
    
    # 计算缺失率
    missing_rates = {}
    for category, count in category_counts.items():
        # 将样本数量映射到[0, max_missrate]区间内，以确保样本数量最多的类别有最高的缺失率
        rate = (count - min_count) / (max_count - min_count) * max_missrate
        missing_rates[category] = round(rate, 2)
    
    print("missing_rates is", missing_rates)

    return missing_rates


def get_train_sn(view_num, alldata_len, y, max_missrate):
    """
    根据类别的样本数量动态调整缺失率，并控制误差。
    """

    unique, counts = np.unique(y, return_counts=True)
    category_counts = dict(zip(unique, counts))
    missing_rates = calculate_missing_rates(category_counts, max_missrate)
    
    # 初始化Sn矩阵
    Sn = np.zeros((alldata_len, view_num))
    enc = OneHotEncoder(categories=[np.arange(view_num)])
    for category, missing_rate in missing_rates.items():
        one_rate = 1 - missing_rate
        indices = np.where(y == category)[0]  # 当前类别的所有样本索引 
        if one_rate == 1: # 不缺失 生成全1矩阵
            temp_matrix = randint(1, 2, size=(len(indices), view_num))
            Sn[indices, :] = temp_matrix
        else:
            enc = OneHotEncoder(categories=[np.arange(view_num)])

            # Step 1: Ensure each sample has at least one view preserved
            random_views = np.random.randint(0, view_num, size=(len(indices), 1))
            view_preserve = enc.fit_transform(random_views).toarray()

            # Step 2: Calculate the number of additional views to preserve
            total_elements = view_num * len(indices)
            one_num = int(total_elements * one_rate)
            additional_ones = one_num - len(indices)  # 已经保留了每个样本的一个视图

            # Step 3: Create a flat array of zeros with the required number of additional ones
            flat_matrix = np.zeros(total_elements - len(indices)) # 还需要添加的1的个数
            flat_matrix[:additional_ones] = 1
            np.random.shuffle(flat_matrix)

            # Step 4: Reshape the flat array and add it to view_preserve
            flat_matrix = flat_matrix.reshape(len(indices), view_num - 1)
            matrix_iter = np.zeros((len(indices), view_num))
            for i in range(len(indices)):
                preserved_index = np.argmax(view_preserve[i])
                other_indices = list(range(view_num))
                other_indices.remove(preserved_index)
                matrix_iter[i, other_indices] = flat_matrix[i]

            # Combine view_preserve and matrix_iter
            matrix = ((view_preserve + matrix_iter) > 0).astype(np.int_)
            actual_one_rate  = np.sum(matrix) / (view_num * len(indices))
            error = abs(one_rate - actual_one_rate )
            Sn[indices, :] = matrix
            
    return Sn

    

def get_test_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number 视图数量
    :param alldata_len:number of samples 样本数量
    :param missing_rate:
    :return:Sn
    """
    # one_rate = 1-missing_rate
    # if one_rate <= (1 / view_num): # 缺失率大于1-1/view_num 随机保存一个视图
    #     enc = OneHotEncoder()
    #     view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
    #     return view_preserve
    # error = 1
    # if one_rate == 1: # 不缺失 生成全1矩阵
    #     matrix = randint(1, 2, size=(alldata_len, view_num))
    #     return matrix
    # while error >= 0.005:
    #     enc = OneHotEncoder(categories=[np.arange(view_num)])
    #     view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
    #     one_num = view_num * alldata_len * one_rate - alldata_len # 每个样本随机保存一个视图后，多余的视图
    #     ratio = one_num / (view_num * alldata_len) # 除了每个样本保存一个视图后，其余生成的矩阵中1的比例
    #     matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
    #     a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int_))
    #     one_num_iter = one_num / (1 - a / one_num)
    #     ratio = one_num_iter / (view_num * alldata_len)
    #     matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
    #     matrix = ((matrix_iter + view_preserve) > 0).astype(np.int_)
    #     actual_one_rate  = np.sum(matrix) / (view_num * alldata_len)
    #     error = abs(one_rate - actual_one_rate )
    # return matrix # 返回的矩阵中1的比例接近于缺失率 大小:(样本数，视图数) 0表示缺失 1表示不缺失
    one_rate = 1 - missing_rate
    enc = OneHotEncoder(categories=[np.arange(view_num)])

    # Step 1: Ensure each sample has at least one view preserved
    random_views = np.random.randint(0, view_num, size=(alldata_len, 1))
    view_preserve = enc.fit_transform(random_views).toarray()

    # Step 2: Calculate the number of additional views to preserve
    total_elements = view_num * alldata_len
    one_num = int(total_elements * one_rate)
    additional_ones = one_num - alldata_len  # 已经保留了每个样本的一个视图

    # Step 3: Create a flat array of zeros with the required number of additional ones
    flat_matrix = np.zeros(total_elements - alldata_len) # 还需要添加的1的个数
    flat_matrix[:additional_ones] = 1
    np.random.shuffle(flat_matrix)

    # Step 4: Reshape the flat array and add it to view_preserve
    flat_matrix = flat_matrix.reshape(alldata_len, view_num - 1)
    matrix_iter = np.zeros((alldata_len, view_num))
    for i in range(alldata_len):
        preserved_index = np.argmax(view_preserve[i])
        other_indices = list(range(view_num))
        other_indices.remove(preserved_index)
        matrix_iter[i, other_indices] = flat_matrix[i]

    # Combine view_preserve and matrix_iter
    matrix = ((view_preserve + matrix_iter) > 0).astype(np.int_)
    actual_one_rate  = np.sum(matrix) / (view_num * alldata_len)
    error = abs(one_rate - actual_one_rate )

    return matrix  # 返回的矩阵中1的比例接近于缺失率 大小:(样本数，视图数) 0表示缺失 1表示不缺失

def save_Sn(Sn,path):
    np.savetxt(path, Sn, delimiter=',')


def load_Sn(str_name):
    return np.loadtxt(str_name + '.csv', delimiter=',')


def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number 视图数量
    :param alldata_len:number of samples 样本数量
    :param missing_rate:
    :return:Sn
    """
    one_rate = 1-missing_rate
    if one_rate <= (1 / view_num): # 缺失率大于1-1/view_num 随机保存一个视图
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1: # 不缺失 生成全1矩阵
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len # 每个样本随机保存一个视图后，多余的视图
        ratio = one_num / (view_num * alldata_len) # 除了每个样本保存一个视图后，其余生成的矩阵中1的比例
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int_))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int_)
        actual_one_rate  = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - actual_one_rate )
    return matrix # 返回的矩阵中1的比例接近于缺失率 大小:(样本数，视图数) 0表示缺失 1表示不缺失

# if __name__ == '__main__':
#     sn = get_sn(2,2000,0.8)
#     save_Sn(sn,'./sh.csv')

