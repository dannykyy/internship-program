# %% [markdown]
# # **Homework 1: COVID-19 Cases Prediction (LSTM-based Regression)**

# %% [markdown]
# Objectives:
# * Solve a regression problem with LSTM and deep neural networks.
# * Handle time series data with static features.
# * Implement attention mechanism for feature importance learning.

# %% [markdown]
# # Import packages

# %%
# Numerical Operations
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 彻底禁用 TensorFlow 日志
# os.environ['TENSORBOARD_DISABLE_TF'] = '1'  # 阻止 TensorFlow 劫持 TensorBoard
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()  # 禁用 TensorFlow 的自动行为
import math
import numpy as np
import matplotlib.pyplot as plt
# Reading/Writing Data
import pandas as pd
import csv

# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
from torch.utils.tensorboard import SummaryWriter
# %% [markdown]
# # Some Utility Functions

# %%
def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_test_split(data_set, valid_ratio, test_ratio, seed):
    '''Split provided training data into training set, validation set and test set'''
    # First split into train+valid and test
    test_set_size = int(test_ratio * len(data_set))
    train_valid_size = len(data_set) - test_set_size
    train_valid_set, test_set = random_split(data_set, [train_valid_size, test_set_size], 
                                           generator=torch.Generator().manual_seed(seed))
    
    # Then split train+valid into train and valid
    valid_set_size = int(valid_ratio * len(train_valid_set))
    train_set_size = len(train_valid_set) - valid_set_size
    train_set, valid_set = random_split(train_valid_set, [train_set_size, valid_set_size], 
                                      generator=torch.Generator().manual_seed(seed))
    
    return np.array(train_set), np.array(valid_set), np.array(test_set)

def predict(test_loader, model, device):
    model.eval()
    preds = []
    for time_seq, static_feat in tqdm(test_loader):
        time_seq = time_seq.to(device)
        static_feat = static_feat.to(device)
        with torch.no_grad():
            pred = model(time_seq, static_feat)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

# %% [markdown]
# # Dataset

# %%
class COVID19Dataset(Dataset):
    '''
    Dataset for COVID-19 prediction with time series and static features
    '''
    def __init__(self, time_series, static_features, y=None):
        self.time_series = torch.FloatTensor(time_series)
        self.static_features = torch.FloatTensor(static_features)
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)

    def __getitem__(self, idx):
        if self.y is None:
            return self.time_series[idx], self.static_features[idx]
        else:
            return self.time_series[idx], self.static_features[idx], self.y[idx]

    def __len__(self):
        return len(self.time_series)

# %% [markdown]
# # Neural Network Model with RNN and Attention

# %%
class AttentionModule(nn.Module):
    """学习时间序列和静态特征的重要性权重"""
    def __init__(self, rnn_hidden_dim, static_dim):
        super(AttentionModule, self).__init__()
        self.rnn_attention = nn.Sequential(
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(rnn_hidden_dim//2 , 1),
            nn.Sigmoid()
        )
        self.static_attention = nn.Sequential(
            nn.Linear(static_dim, static_dim // 2),
            nn.ReLU(),
            nn.Linear(static_dim//2 , 1),
            nn.Sigmoid()
        )
    
    def forward(self, rnn_output, static_output):
        rnn_weight = self.rnn_attention(rnn_output)
        static_weight = self.static_attention(static_output)
        
        # 归一化权重
        total_weight = rnn_weight + static_weight
        rnn_weight = rnn_weight / total_weight
        static_weight = static_weight / total_weight
        
        return rnn_weight, static_weight

class COVID19_RNN_Model(nn.Module):
    def __init__(self, time_features, static_features, rnn_hidden_dim=64, rnn_layers=2, final_layers_1_dim=128, final_layers_2_dim=64):
        super(COVID19_RNN_Model, self).__init__()
        
        # 时间序列分支
        self.rnn = nn.RNN(
            input_size=time_features,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=config['dropout'] if rnn_layers > 1 else 0
        )
        
        # 静态特征分支 - 全连接层
        # self.static_branch = nn.Identity()  # 直接返回输入不做任何变换
        self.static_branch = nn.Sequential(
            nn.Linear(static_features, rnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0)
        )
        
        # RNN输出处理
        self.rnn_fc=nn.Identity()
        
        # 注意力机制
        self.attention = AttentionModule(rnn_hidden_dim, rnn_hidden_dim)
        
        # 合并后的全连接层
        self.final_layers = nn.Sequential(
            nn.Linear(rnn_hidden_dim*2, final_layers_1_dim),
            nn.BatchNorm1d(final_layers_1_dim),
            nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.Linear(final_layers_1_dim, final_layers_2_dim),
            # nn.ReLU(),
            nn.Linear(final_layers_1_dim, 1)
        )

    def forward(self, time_seq, static_feat):
        # 时间序列分支
        rnn_out, hidden = self.rnn(time_seq)
        # 取最后一个时间步的输出
        rnn_final = self.rnn_fc(rnn_out[:, -1, :])
        
        # 静态特征分支
        static_out = self.static_branch(static_feat)
        
        # 注意力机制学习权重
        rnn_weight, static_weight = self.attention(rnn_final, static_out)
        
        # 加权合并
        weighted_rnn = rnn_final * rnn_weight
        weighted_static = static_out * static_weight
        
        # 合并特征
        combined = torch.cat([weighted_rnn, weighted_static], dim=1)
        
        # 最终预测
        output = self.final_layers(combined)
        return output.squeeze(1)


# %% [markdown]
# # Data Preprocessing

# %%
def prepare_data(train_data, valid_data, test_data):
    '''
    准备时间序列数据和静态特征
    数据结构：第1列id，第2-38列地区独热编码，第39列开始是5天的时间序列特征
    '''
    # 分离特征和标签
    y_train = train_data[:, -1]  # 最后一列是target
    y_valid = valid_data[:, -1]
    y_test= test_data[:, -1]
    # 移除id列和target列
    train_features = train_data[:, 1:-1]  # 移除id和target
    valid_features = valid_data[:, 1:-1]
    test_features = test_data[:, 1:-1]
    
    # 静态特征：地区独热编码 (列 1-37, 在移除id后是 0-36)
    static_train = train_features[:, :37]
    static_valid = valid_features[:, :37]
    static_test = test_features[:, :37]
    
    # 时间序列特征：连续5天的特征 (从第38列开始，在移除id后是从第37列开始)
    time_features = train_features[:, 37:]
    time_valid_features = valid_features[:, 37:]
    time_test_features = test_features[:, 37:]
    
    # 计算每天的特征数量
    # 总共有117列，减去1列id和37列地区 = 79列时间序列特征
    # 79列 / 5天 = 15.8，实际应该是16个特征每天，最后一天少一个(tested_positive)
    features_per_day = 16
    
    # 重塑时间序列数据为 (samples, timesteps, features)
    def reshape_time_series(data, is_train=True):
        samples = data.shape[0]
        timesteps = 5
        # 为了保持一致性，我们为第5天的缺失特征填充0
        padded_data = np.zeros((samples, timesteps * features_per_day))
        padded_data[:, :data.shape[1]] = data
        reshaped = padded_data.reshape(samples, timesteps, features_per_day)
        return reshaped
    
    # 重塑时间序列数据
    time_series_train = reshape_time_series(time_features, is_train=True)
    time_series_valid = reshape_time_series(time_valid_features, is_train=True)
    time_series_test = reshape_time_series(time_test_features, is_train=False)
    
    return (time_series_train, static_train, time_series_valid, static_valid, 
            time_series_test, static_test, y_train, y_valid, y_test)


# %% [markdown]
# # Training Loop

# %%
def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=config['learning_rate'])
    train_losses = []
    valid_losses = []
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    # os.makedirs('./torch_logs', exist_ok=True)
    # writer = SummaryWriter(log_dir='./torch_logs')

    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for time_seq, static_feat, y in train_pbar:
            optimizer.zero_grad()
            time_seq = time_seq.to(device)
            static_feat = static_feat.to(device)
            y = y.to(device)
            
            pred = model(time_seq, static_feat)
            loss = criterion(pred, y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            step += 1
            loss_record.append(loss.detach().item())

            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
        # writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()
        loss_record = []
        for time_seq, static_feat, y in valid_loader:
            time_seq = time_seq.to(device)
            static_feat = static_feat.to(device)
            y = y.to(device)
            
            with torch.no_grad():
                pred = model(time_seq, static_feat)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        # writer.add_scalar('Loss/valid', mean_valid_loss, step)
        train_losses.append(mean_train_loss)
        valid_losses.append(mean_valid_loss)
        
        # 更新学习率
        scheduler.step(mean_valid_loss)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            # 在函数返回前添加可视化（训练结束后）
            plt.figure(figsize=(8, 5))
            plt.plot(train_losses, 'b-', label='Train Loss')
            plt.plot(valid_losses, 'r-', label='Valid Loss')
            
            # 标注最佳验证点
            best_epoch = np.argmin(valid_losses)
            plt.scatter(best_epoch, valid_losses[best_epoch], 
                        color='gold', s=100, label=f'Best (Epoch {best_epoch+1})')
            
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.show()
            # 自动保存图像
            os.makedirs('./plots', exist_ok=True)
            plt.savefig('./plots/loss_curves.png')
            
            # plt.close()
            return
    # 在函数返回前添加可视化（训练结束后）
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(valid_losses, 'r-', label='Valid Loss')
    
    # 标注最佳验证点
    best_epoch = np.argmin(valid_losses)
    plt.scatter(best_epoch, valid_losses[best_epoch], 
                color='gold', s=100, label=f'Best (Epoch {best_epoch+1})')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    # 自动保存图像
    os.makedirs('./plots', exist_ok=True)
    plt.savefig('./plots/loss_curves.png')
    # plt.close()
    # writer.close()
    return


# %% [markdown]
# # Configurations

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,
    'valid_ratio': 0.15,  # Reduced from 0.2 to make room for test set
    'test_ratio': 0.15,   # Added test ratio
    'n_epochs': 200,
    'batch_size': 32,
    'learning_rate': 0.003108968287249924,
    'early_stop': 40,
    'save_path': './models/lstm_model.ckpt',
    'hidden_size': 60,        # LSTM隐藏层维度
    'num_layers': 1,           # LSTM堆叠层数
    'dropout': 0.2,             # 可选：添加dropout比例
    'final_layers_1_dim': 114, # 全连接层1维度
    # 'final_layers_2_dim': 0  # 全连接层2维度
}

# %% [markdown]
# # Dataloader

# %%
# Set seed for reproducibility
same_seed(config['seed'])

# 读取数据
script_dir = os.path.dirname(os.path.abspath(__file__))
train_data = pd.read_csv(os.path.join(script_dir, 'covid.train.csv')).values

# Split into train, valid, and test sets
train_data, valid_data, test_data = train_valid_test_split(
    train_data, 
    valid_ratio=config['valid_ratio'],
    test_ratio=config['test_ratio'],
    seed=config['seed']
)

print(f"""train_data size: {train_data.shape}
valid_data size: {valid_data.shape}  
test_data size: {test_data.shape}""")

# 准备数据
(time_series_train, static_train, time_series_valid, static_valid, 
 time_series_test, static_test, y_train, y_valid,y_test) = prepare_data(train_data, valid_data, test_data)

# 数据归一化
# 时间序列数据归一化
time_series_train_flat = time_series_train.reshape(-1, time_series_train.shape[-1])
time_min, time_max = time_series_train_flat.min(axis=0), time_series_train_flat.max(axis=0)
time_range = time_max - time_min
time_range[time_range == 0] = 1  # 避免除零

time_series_train_norm = (time_series_train - time_min) / time_range
time_series_valid_norm = (time_series_valid - time_min) / time_range
time_series_test_norm = (time_series_test - time_min) / time_range

# 静态特征归一化
static_min, static_max = static_train.min(axis=0), static_train.max(axis=0)
static_range = static_max - static_min
static_range[static_range == 0] = 1  # 避免除零

static_train_norm = (static_train - static_min) / static_range
static_valid_norm = (static_valid - static_min) / static_range
static_test_norm = (static_test - static_min) / static_range

print(f'Time series shape: {time_series_train_norm.shape}')
print(f'Static features shape: {static_train_norm.shape}')

# 创建数据集
train_dataset = COVID19Dataset(time_series_train_norm, static_train_norm, y_train)
valid_dataset = COVID19Dataset(time_series_valid_norm, static_valid_norm, y_valid)
test_dataset = COVID19Dataset(time_series_test_norm, static_test_norm)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

# %% [markdown]
# # Start training!

# %%
# 创建模型
time_features = time_series_train_norm.shape[2]  # 每个时间步的特征数
static_features = static_train_norm.shape[1]     # 静态特征数

model = COVID19_RNN_Model(
    time_features=time_features,
    static_features=static_features,
    rnn_hidden_dim=config['hidden_size'],  # 从config中获取hidden_size
    rnn_layers=config['num_layers'],       # 同样从config中获取num_layers
    final_layers_1_dim=config['final_layers_1_dim'],  # 从config中获取final_layers_1_dim
    # final_layers_2_dim=0  # 从config中获取final_layers_2_dim
).to(device)

print(f"Model created with {time_features} time features and {static_features} static features")
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

# 开始训练
trainer(train_loader, valid_loader, model, config, device)


# %% [markdown]
# # Testing

# %%
def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

# 加载最佳模型并预测
model = COVID19_RNN_Model(
    time_features=time_features,
    static_features=static_features,
    rnn_hidden_dim=config['hidden_size'],  # 从config中获取hidden_size
    rnn_layers=config['num_layers'] ,      # 同样从config中获取num_layers
    final_layers_1_dim=config['final_layers_1_dim'],  # 从config中获取final_layers_1_dim
    # final_layers_2_dim=config['final_layers_2_dim']  # 从config中获取final_layers_2_dim
).to(device)
model.load_state_dict(torch.load(config['save_path'], weights_only=True))

# 计算训练集、验证集和测试集的指标
def calculate_metrics(model, time_series, static, y_true, device='cpu', set_name='Dataset'):
    """
    计算模型在给定数据上的多项指标并打印结果：
    - MSE: 均方误差
    - MAE: 平均绝对误差
    - MAPE: 平均绝对百分比误差（%）
    - R²: 决定系数
    
    参数:
        model: PyTorch模型
        time_series: 时间序列数据 [num_samples, seq_len, features]
        static: 静态特征数据 [num_samples, static_features]
        y_true: 真实值 [num_samples, ]
        device: 计算设备
        set_name: 数据集名称（用于打印标识）
    返回:
        指标字典 {'mse': float, 'mae': float, 'mape': float, 'r2': float}
    """
    model.eval()
    with torch.no_grad():
        # 转换数据为张量
        time_series_tensor = torch.FloatTensor(time_series).to(device)
        static_tensor = torch.FloatTensor(static).to(device)
        y_true_tensor = torch.FloatTensor(y_true).to(device)
        
        # 获取预测结果
        outputs = model(time_series_tensor, static_tensor)
        y_pred = outputs.cpu().numpy().flatten()
        y_true = y_true.flatten()
        
        # 计算各项指标
        # mse = np.mean((y_true - y_pred) ** 2)
        criterion = nn.MSELoss()  # 默认返回均值损失（reduction='mean'）
        mse = criterion(outputs, y_true_tensor)  # 注意参数顺序：pred在前，true在后
        mae = np.mean(np.abs(y_true - y_pred))
        
        # MAPE计算（添加epsilon防止零除）
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
        
        # R²计算
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + epsilon))
    
    # 格式化打印结果
    print(f"\n{' ' + set_name + ' Evaluation ':=^60}")
    print(f"{'MSE (Mean Squared Error)':<30}: {mse:.6f}")
    print(f"{'MAE (Mean Absolute Error)':<30}: {mae:.6f}")
    print(f"{'MAPE (%)':<30}: {mape:.2f}%")
    print(f"{'R-squared (R²)':<30}: {r2:.4f}")
    print("=" * 60)
    
    return {'mse': mse, 'mae': mae, 'mape': mape, 'r2': r2}

# 计算各数据集指标
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_metrics = calculate_metrics(model, time_series_train_norm, static_train, y_train, device, set_name='训练集')
valid_metrics = calculate_metrics(model, time_series_valid_norm, static_valid, y_valid, device, set_name='验证集')
test_metrics = calculate_metrics(model, time_series_test_norm, static_test, y_test, device, set_name='测试集')

# # 预测测试集并保存结果
# preds = predict(test_loader, model, device)
# save_pred(preds, 'rnn_pred.csv')

# print("预测完成！结果保存在 lstm_pred.csv")


# %% [markdown]
# # Model Analysis

# %%
# 分析模型学到的注意力权重
def analyze_attention_weights(model, data_loader, device, num_samples=100):
    model.eval()
    rnn_weights = []
    static_weights = []
    
    with torch.no_grad():
        for i, (time_seq, static_feat, _) in enumerate(data_loader):
            if i * data_loader.batch_size >= num_samples:
                break
                
            time_seq = time_seq.to(device)
            static_feat = static_feat.to(device)
            
            # 获取LSTM输出
            rnn_out, _ = model.rnn(time_seq)
            rnn_final = model.rnn_fc(rnn_out[:, -1, :])
            
            # 获取静态特征输出
            static_out = model.static_branch(static_feat)
            
            # 获取注意力权重
            rnn_weight, static_weight = model.attention(rnn_final, static_out)
            
            rnn_weights.extend(rnn_weight.cpu().numpy())
            static_weights.extend(static_weight.cpu().numpy())
    
    rnn_weights = np.array(rnn_weights)
    static_weights = np.array(static_weights)
    
    print(f"RNN分支平均权重: {rnn_weights.mean():.4f} ± {rnn_weights.std():.4f}")
    print(f"静态特征分支平均权重: {static_weights.mean():.4f} ± {static_weights.std():.4f}")
    
    return rnn_weights, static_weights

# 分析注意力权重
rnn_weights, static_weights = analyze_attention_weights(model, valid_loader, device)

# %% [markdown]
# # Reference
# This code is based on the original COVID-19 prediction framework and enhanced with LSTM for time series processing and attention mechanism for feature importance learning.