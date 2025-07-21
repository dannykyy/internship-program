import os
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Must be placed before import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pandas as pd
import csv
import threading
import time
from tqdm import tqdm
import gradio as gr
from gradio import Timer
# PyTorch related
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# Set matplotlib Chinese display
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Global variable to control training status
training_state = {
    'is_training': False,
    'should_stop': False,
    'should_pause': False,
    'current_epoch': 0,
    'total_epochs': 0,
    'train_losses': [],
    'valid_losses': [],
    'best_epoch': 0,
    'attention_weights': {'rnn': [], 'static': []}
}


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
    test_set_size = int(test_ratio * len(data_set))
    train_valid_size = len(data_set) - test_set_size
    train_valid_set, test_set = random_split(data_set, [train_valid_size, test_set_size],
                                             generator=torch.Generator().manual_seed(seed))

    valid_set_size = int(valid_ratio * len(train_valid_set))
    train_set_size = len(train_valid_set) - valid_set_size
    train_set, valid_set = random_split(train_valid_set, [train_set_size, valid_set_size],
                                        generator=torch.Generator().manual_seed(seed))

    return np.array(train_set), np.array(valid_set), np.array(test_set)


class COVID19Dataset(Dataset):
    '''Dataset for COVID-19 prediction with time series and static features'''

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


class AttentionModule(nn.Module):
    """Learn importance weights for time series and static features"""

    def __init__(self, rnn_hidden_dim, static_dim):
        super(AttentionModule, self).__init__()
        self.rnn_attention = nn.Sequential(
            nn.Linear(rnn_hidden_dim, rnn_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(rnn_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.static_attention = nn.Sequential(
            nn.Linear(static_dim, static_dim // 2),
            nn.ReLU(),
            nn.Linear(static_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, rnn_output, static_output):
        rnn_weight = self.rnn_attention(rnn_output)
        static_weight = self.static_attention(static_output)

        total_weight = rnn_weight + static_weight
        rnn_weight = rnn_weight / total_weight
        static_weight = static_weight / total_weight

        return rnn_weight, static_weight


class COVID19_RNN_Model(nn.Module):
    def __init__(self, time_features, static_features, rnn_hidden_dim=64, rnn_layers=2, final_layers_1_dim=128,
                 dropout=0.2):
        super(COVID19_RNN_Model, self).__init__()

        self.rnn = nn.RNN(
            input_size=time_features,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0
        )

        self.static_branch = nn.Sequential(
            nn.Linear(static_features, rnn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0)
        )

        self.rnn_fc = nn.Identity()
        self.attention = AttentionModule(rnn_hidden_dim, rnn_hidden_dim)

        self.final_layers = nn.Sequential(
            nn.Linear(rnn_hidden_dim * 2, final_layers_1_dim),
            nn.BatchNorm1d(final_layers_1_dim),
            nn.ReLU(),
            nn.Linear(final_layers_1_dim, 1)
        )

    def forward(self, time_seq, static_feat):
        rnn_out, hidden = self.rnn(time_seq)
        rnn_final = self.rnn_fc(rnn_out[:, -1, :])
        static_out = self.static_branch(static_feat)

        rnn_weight, static_weight = self.attention(rnn_final, static_out)

        weighted_rnn = rnn_final * rnn_weight
        weighted_static = static_out * static_weight

        combined = torch.cat([weighted_rnn, weighted_static], dim=1)
        output = self.final_layers(combined)
        return output.squeeze(1)


def prepare_data(train_data, valid_data, test_data):
    '''Prepare time series data and static features'''
    y_train = train_data[:, -1]
    y_valid = valid_data[:, -1]
    y_test = test_data[:, -1]

    train_features = train_data[:, 1:-1]
    valid_features = valid_data[:, 1:-1]
    test_features = test_data[:, 1:-1]

    static_train = train_features[:, :37]
    static_valid = valid_features[:, :37]
    static_test = test_features[:, :37]

    time_features = train_features[:, 37:]
    time_valid_features = valid_features[:, 37:]
    time_test_features = test_features[:, 37:]

    features_per_day = 16

    def reshape_time_series(data, is_train=True):
        samples = data.shape[0]
        timesteps = 5
        padded_data = np.zeros((samples, timesteps * features_per_day))
        padded_data[:, :data.shape[1]] = data
        reshaped = padded_data.reshape(samples, timesteps, features_per_day)
        return reshaped

    time_series_train = reshape_time_series(time_features, is_train=True)
    time_series_valid = reshape_time_series(time_valid_features, is_train=True)
    time_series_test = reshape_time_series(time_test_features, is_train=False)

    return (time_series_train, static_train, time_series_valid, static_valid,
            time_series_test, static_test, y_train, y_valid, y_test)


def create_loss_plot(train_losses, valid_losses, best_epoch, epoch_range=None):
    """Create loss curve plot"""
    if not train_losses or not valid_losses:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, '训练尚未开始', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('训练和验证损失曲线')
        return fig

    fig, ax = plt.subplots(figsize=(6, 4))

    if epoch_range:
        start_epoch, end_epoch = epoch_range
        start_epoch = max(0, int(start_epoch))
        end_epoch = min(len(train_losses) - 1, int(end_epoch))  # Adjust end_epoch to be within bounds

        epochs = list(range(start_epoch, end_epoch + 1))  # +1 to include the end_epoch
        train_subset = train_losses[start_epoch:end_epoch + 1]
        valid_subset = valid_losses[start_epoch:end_epoch + 1]
    else:
        epochs = list(range(len(train_losses)))
        train_subset = train_losses
        valid_subset = valid_losses

    if train_subset and valid_subset and len(epochs) == len(train_subset):  # Ensure data consistency
        ax.plot(epochs, train_subset, 'b-', label='训练损失', linewidth=2)
        ax.plot(epochs, valid_subset, 'r-', label='验证损失', linewidth=2)

        # Plot best epoch only if it's within the current visible range
        if best_epoch < len(valid_losses) and (not epoch_range or (start_epoch <= best_epoch <= end_epoch)):
            ax.scatter(best_epoch, valid_losses[best_epoch],
                       color='gold', s=100, label=f'最佳点 (Epoch {best_epoch + 1})', zorder=5)

    ax.set_title('训练和验证损失曲线')
    ax.set_xlabel('轮次 (Epoch)')
    ax.set_ylabel('损失')
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_model_comparison_plot(current_mse=None):
    """Create model comparison bar chart"""
    fig, ax = plt.subplots(figsize=(6, 4))

    models = ['朴素预测', '加权滑动平均', 'RNN模型']
    mse_values = [1.2192, 2.0296, current_mse if current_mse else 0]
    colors = ['#ff9999', '#66b3ff', '#99ff99']

    bars = ax.bar(models, mse_values, color=colors, alpha=0.8)

    for bar, value in zip(bars, mse_values):
        if value > 0:
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')

    ax.set_title('模型验证集MSE对比')
    ax.set_ylabel('MSE')
    ax.set_ylim(0, max(3, max(mse_values) * 1.1))
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def create_attention_bar(rnn_weight, static_weight):
    """Create attention weights bar chart"""
    fig, ax = plt.subplots(figsize=(6, 1.5))

    if rnn_weight is None or static_weight is None:
        ax.text(0.5, 0.5, '训练尚未开始', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('平均注意力权重分布')
        return fig

    categories = ['注意力权重分布']
    width = 0.8

    ax.barh(categories, [rnn_weight], width, label=f'RNN权重 ({rnn_weight:.3f})',
            color='skyblue', alpha=0.8)
    ax.barh(categories, [static_weight], width, left=[rnn_weight],
            label=f'静态特征权重 ({static_weight:.3f})', color='lightcoral', alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_title('平均注意力权重分布')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig


def create_scatter_plot(y_true, y_pred):
    """Create regression scatter plot"""
    if y_true is None or y_pred is None or len(y_true) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, '暂无预测数据', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('预测vs真实值散点图')
        return fig

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(y_true, y_pred, alpha=0.6, s=20)

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='理想预测线')

    ax.set_xlabel('真实值')
    ax.set_ylabel('预测值')
    ax.set_title('预测vs真实值散点图')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def calculate_metrics_batch(model, time_series, static, y_true, device, batch_size=64):
    """Calculate metrics in batches to avoid memory overflow"""
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(time_series), batch_size):
            end_idx = min(i + batch_size, len(time_series))

            time_batch = torch.FloatTensor(time_series[i:end_idx]).to(device)
            static_batch = torch.FloatTensor(static[i:end_idx]).to(device)

            outputs = model(time_batch, static_batch)
            all_preds.append(outputs.cpu().numpy())

    y_pred = np.concatenate(all_preds).flatten()
    y_true = y_true.flatten()

    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))

    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))

    return {'MSE': mse, 'MAE': mae, 'MAPE': mape, 'R²': r2}, y_pred


def get_attention_weights(model, data_loader, device, num_samples=100):
    """Get attention weights"""
    model.eval()
    rnn_weights = []
    static_weights = []

    with torch.no_grad():
        sample_count = 0
        for time_seq, static_feat, _ in data_loader:
            if sample_count >= num_samples:
                break

            time_seq = time_seq.to(device)
            static_feat = static_feat.to(device)

            rnn_out, _ = model.rnn(time_seq)
            rnn_final = model.rnn_fc(rnn_out[:, -1, :])
            static_out = model.static_branch(static_feat)

            rnn_weight, static_weight = model.attention(rnn_final, static_out)

            rnn_weights.extend(rnn_weight.cpu().numpy())
            static_weights.extend(static_weight.cpu().numpy())

            sample_count += len(time_seq)

    if rnn_weights and static_weights:
        return np.mean(rnn_weights), np.mean(static_weights)
    return None, None


def training_thread(model, train_loader, valid_loader, config, device, data_for_viz):
    """Training thread function"""
    global training_state

    training_state['is_training'] = True
    training_state['should_stop'] = False
    training_state['should_pause'] = False
    training_state['train_losses'] = []
    training_state['valid_losses'] = []
    training_state['current_epoch'] = 0
    training_state['total_epochs'] = config['n_epochs']

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss = math.inf
    early_stop_count = 0

    for epoch in range(config['n_epochs']):
        if training_state['should_stop']:
            break

        while training_state['should_pause']:
            time.sleep(0.1)
            if training_state['should_stop']:
                break

        if training_state['should_stop']:
            break

        # Training phase
        model.train()
        train_loss_record = []

        for time_seq, static_feat, y in train_loader:
            if training_state['should_stop']:
                break

            optimizer.zero_grad()
            time_seq = time_seq.to(device)
            static_feat = static_feat.to(device)
            y = y.to(device)

            model.train()
            pred = model(time_seq, static_feat)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_record.append(loss.detach().item())

        if training_state['should_stop']:
            break

        mean_train_loss = sum(train_loss_record) / len(train_loss_record)

        # Validation phase
        model.eval()
        valid_loss_record = []
        with torch.no_grad():
            for time_seq, static_feat, y in valid_loader:
                time_seq = time_seq.to(device)
                static_feat = static_feat.to(device)
                y = y.to(device)

                pred = model(time_seq, static_feat)
                loss = criterion(pred, y)
                valid_loss_record.append(loss.item())

        mean_valid_loss = sum(valid_loss_record) / len(valid_loss_record)

        # Update training state
        training_state['train_losses'].append(mean_train_loss)
        training_state['valid_losses'].append(mean_valid_loss)
        training_state['current_epoch'] = epoch + 1

        # Get attention weights
        rnn_weight, static_weight = get_attention_weights(model, valid_loader, device)
        training_state['attention_weights']['rnn'].append(rnn_weight if rnn_weight else 0)
        training_state['attention_weights']['static'].append(static_weight if static_weight else 0)

        scheduler.step(mean_valid_loss)

        # Early stopping check
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            training_state['best_epoch'] = epoch
            early_stop_count = 0
            # Save model
            os.makedirs('./models', exist_ok=True)
            torch.save(model.state_dict(), config['save_path'])
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            break

    training_state['is_training'] = False


def create_gradio_interface():
    # Read data (only once)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        train_data = pd.read_csv('covid.train.csv').values
    except:
        print("Warning: covid.train.csv not found, using sample data")
        train_data = np.random.randn(1000, 120)

    global_data = {'train_data': train_data, 'model': None, 'data_loaders': None, 'processed_data': None}

    def start_training(seed, valid_ratio, test_ratio, n_epochs, batch_size, learning_rate,
                       early_stop, hidden_size, num_layers, dropout, final_layers_1_dim):
        global training_state

        if training_state['is_training']:
            return "训练正在进行中，请先停止当前训练"

        config = {
            'seed': int(seed),
            'valid_ratio': valid_ratio,
            'test_ratio': test_ratio,
            'n_epochs': int(n_epochs),
            'batch_size': int(batch_size),
            'learning_rate': learning_rate,
            'early_stop': int(early_stop),
            'save_path': './models/lstm_model.ckpt',
            'hidden_size': int(hidden_size),
            'num_layers': int(num_layers),
            'dropout': dropout,
            'final_layers_1_dim': int(final_layers_1_dim)
        }

        try:
            same_seed(config['seed'])

            train_data_split, valid_data_split, test_data_split = train_valid_test_split(
                global_data['train_data'], config['valid_ratio'], config['test_ratio'], config['seed']
            )

            (time_series_train, static_train, time_series_valid, static_valid,
             time_series_test, static_test, y_train, y_valid, y_test) = prepare_data(
                train_data_split, valid_data_split, test_data_split
            )

            time_series_train_flat = time_series_train.reshape(-1, time_series_train.shape[-1])
            time_min, time_max = time_series_train_flat.min(axis=0), time_series_train_flat.max(axis=0)
            time_range = time_max - time_min
            time_range[time_range == 0] = 1

            time_series_train_norm = (time_series_train - time_min) / time_range
            time_series_valid_norm = (time_series_valid - time_min) / time_range
            time_series_test_norm = (time_series_test - time_min) / time_range

            static_min, static_max = static_train.min(axis=0), static_train.max(axis=0)
            static_range = static_max - static_min
            static_range[static_range == 0] = 1

            static_train_norm = (static_train - static_min) / static_range
            static_valid_norm = (static_valid - static_min) / static_range
            static_test_norm = (static_test - static_min) / static_range

            train_dataset = COVID19Dataset(time_series_train_norm, static_train_norm, y_train)
            valid_dataset = COVID19Dataset(time_series_valid_norm, static_valid_norm, y_valid)

            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            time_features = time_series_train_norm.shape[2]
            static_features = static_train_norm.shape[1]

            model = COVID19_RNN_Model(
                time_features=time_features,
                static_features=static_features,
                rnn_hidden_dim=config['hidden_size'],
                rnn_layers=config['num_layers'],
                final_layers_1_dim=config['final_layers_1_dim'],
                dropout=config['dropout']
            ).to(device)

            global_data['model'] = model
            global_data['data_loaders'] = (train_loader, valid_loader)
            global_data['processed_data'] = {
                'time_series_train_norm': time_series_train_norm,
                'static_train_norm': static_train_norm,
                'time_series_valid_norm': time_series_valid_norm,
                'static_valid_norm': static_valid_norm,
                'time_series_test_norm': time_series_test_norm,
                'static_test_norm': static_test_norm,
                'y_train': y_train,
                'y_valid': y_valid,
                'y_test': y_test
            }

            data_for_viz = (time_series_valid_norm, static_valid_norm, y_valid)
            thread = threading.Thread(target=training_thread,
                                      args=(model, train_loader, valid_loader, config, device, data_for_viz))
            thread.daemon = True
            thread.start()

            return "训练已开始！"

        except Exception as e:
            return f"启动训练时出错: {str(e)}"

    def stop_training():
        global training_state
        training_state['should_stop'] = True
        training_state['should_pause'] = False
        return "训练停止指令已发送"

    def pause_training():
        global training_state
        if training_state['is_training']:
            training_state['should_pause'] = not training_state['should_pause']
            return "训练已暂停" if training_state['should_pause'] else "训练已恢复"
        return "当前没有正在进行的训练"

    def get_current_status():
        global training_state
        if training_state['is_training']:
            status = f"正在训练 - Epoch {training_state['current_epoch']}/{training_state['total_epochs']}"
            if training_state['should_pause']:
                status += " (已暂停)"
        elif training_state['train_losses']:
            status = f"训练完成 - 总共 {len(training_state['train_losses'])} 个epoch"
        else:
            status = "未开始训练"
        return status

    def update_loss_plot(epoch_range):
        if epoch_range is None:
            range_vals = None
        else:
            range_vals = epoch_range
        return create_loss_plot(training_state['train_losses'],
                                training_state['valid_losses'],
                                training_state['best_epoch'],
                                range_vals)

    def update_model_comparison():
        current_mse = None
        if training_state['valid_losses']:
            current_mse = min(training_state['valid_losses'])
        return create_model_comparison_plot(current_mse)

    def update_attention_bar():
        if training_state['attention_weights']['rnn']:
            rnn_weight = np.mean(training_state['attention_weights']['rnn'][-10:])
            static_weight = np.mean(training_state['attention_weights']['static'][-10:])
        else:
            rnn_weight = static_weight = None
        return create_attention_bar(rnn_weight, static_weight)

    def update_scatter_plot():
        if not global_data['model'] or not global_data['processed_data']:
            return create_scatter_plot(None, None)

        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = global_data['model']
            data = global_data['processed_data']

            _, y_pred = calculate_metrics_batch(
                model, data['time_series_valid_norm'], data['static_valid_norm'],
                data['y_valid'], device
            )
            return create_scatter_plot(data['y_valid'], y_pred)
        except Exception as e:
            print(f"Error updating scatter plot: {e}")
            return create_scatter_plot(None, None)

    def get_metrics_table():
        if not global_data['model'] or not global_data['processed_data'] or not training_state['train_losses']:
            return pd.DataFrame({'指标': ['MSE', 'MAE', 'MAPE (%)', 'R²'],
                                 '训练集': ['-', '-', '-', '-'],
                                 '验证集': ['-', '-', '-', '-'],
                                 '测试集': ['-', '-', '-', '-']})
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = global_data['model']
            data = global_data['processed_data']

            train_metrics, _ = calculate_metrics_batch(
                model, data['time_series_train_norm'], data['static_train_norm'],
                data['y_train'], device
            )

            valid_metrics, _ = calculate_metrics_batch(
                model, data['time_series_valid_norm'], data['static_valid_norm'],
                data['y_valid'], device
            )

            test_metrics, _ = calculate_metrics_batch(
                model, data['time_series_test_norm'], data['static_test_norm'],
                data['y_test'], device
            )

            return pd.DataFrame({
                '指标': ['MSE', 'MAE', 'MAPE (%)', 'R²'],
                '训练集': [f"{train_metrics['MSE']:.4f}", f"{train_metrics['MAE']:.4f}",
                           f"{train_metrics['MAPE']:.2f}", f"{train_metrics['R²']:.4f}"],
                '验证集': [f"{valid_metrics['MSE']:.4f}", f"{valid_metrics['MAE']:.4f}",
                           f"{valid_metrics['MAPE']:.2f}", f"{valid_metrics['R²']:.4f}"],
                '测试集': [f"{test_metrics['MSE']:.4f}", f"{test_metrics['MAE']:.4f}",
                           f"{test_metrics['MAPE']:.2f}", f"{test_metrics['R²']:.4f}"]
            })
        except Exception as e:
            print(f"Error calculating metrics table: {e}")
            return pd.DataFrame({'指标': ['MSE', 'MAE', 'MAPE (%)', 'R²'],
                                 '训练集': ['-', '-', '-', '-'],
                                 '验证集': ['-', '-', '-', '-'],
                                 '测试集': ['-', '-', '-', '-']})

    with gr.Blocks(title="COVID-19 模型训练可视化", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# COVID-19 病例预测模型训练可视化界面")

        with gr.Row():
            with gr.Column(scale=1):  # This column contains Hyperparameters, Model Structure, and now Metrics Table
                gr.Markdown("### 🔧 超参数设置")
                with gr.Group():
                    seed = gr.Number(label="随机种子", value=5201314, precision=0)
                    valid_ratio = gr.Slider(0.1, 0.3, value=0.15, step=0.05, label="验证集比例")
                    test_ratio = gr.Slider(0.1, 0.3, value=0.15, step=0.05, label="测试集比例")
                    n_epochs = gr.Number(label="训练轮数", value=200, precision=0)
                    batch_size = gr.Number(label="批大小", value=32, precision=0)
                    learning_rate = gr.Number(label="学习率", value=0.001, precision=6)
                    early_stop = gr.Number(label="早停轮数", value=40, precision=0)

                # Use a Row to place Model Structure and Metrics side-by-side within this column
                with gr.Row():
                    with gr.Column(scale=1):  # Column for Model Structure
                        gr.Markdown("### 🏗️ 模型结构")
                        with gr.Group():
                            hidden_size = gr.Number(label="RNN隐藏层维度", value=64, precision=0)
                            num_layers = gr.Number(label="RNN层数", value=1, precision=0)
                            dropout = gr.Slider(0.0, 0.5, value=0.2, step=0.1, label="Dropout（RNN层数>1时生效）")
                            final_layers_1_dim = gr.Number(label="全连接层维度", value=128, precision=0)

                    with gr.Column(scale=1):  # Column for Model Performance Metrics
                        gr.Markdown("### 📋 模型性能指标")
                        metrics_table = gr.Dataframe(
                            value=get_metrics_table(),
                            headers=['指标', '训练集', '验证集', '测试集'],
                            datatype=['str', 'str', 'str', 'str'],
                            interactive=False,
                            height=200
                        )

            with gr.Column(
                    scale=2):  # This column contains Training Control, Model Comparison, Attention Bar, Loss Curve, Scatter Plot
                with gr.Row():  # Training control in one row
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎮 训练控制")
                        with gr.Group():
                            start_btn = gr.Button("🚀 开始训练", variant="primary", size="lg")
                            with gr.Row():
                                pause_btn = gr.Button("⏸️ 暂停/恢复", size="sm")
                                stop_btn = gr.Button("🛑 停止训练", variant="stop", size="sm")

                            status_text = gr.Textbox(
                                label="训练状态",
                                value="未开始训练",
                                interactive=False,
                                max_lines=2
                            )
                            result_text = gr.Textbox(
                                label="操作结果",
                                value="等待开始...",
                                interactive=False,
                                max_lines=2
                            )
                    with gr.Column(scale=1):  # Moved Training Progress here
                        gr.Markdown("### 📈 训练进度")
                        with gr.Group():
                            current_epoch_text = gr.Textbox(
                                label="当前轮次",
                                value="0",
                                interactive=False
                            )
                            best_epoch_text = gr.Textbox(
                                label="最佳轮次",
                                value="-",
                                interactive=False
                            )
                            current_loss_text = gr.Textbox(
                                label="当前验证损失",
                                value="-",
                                interactive=False
                            )

                with gr.Row():  # Model comparison and Attention in one row
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 模型MSE对比")
                        model_comparison_plot = gr.Plot(
                            value=create_model_comparison_plot(),
                            show_label=False
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚖️ 注意力权重分布")
                        attention_bar = gr.Plot(
                            value=create_attention_bar(None, None),
                            show_label=False
                        )
                gr.HTML("""
                                <style>
                                #hidden_epoch_slider { display: none !important; }
                                </style>
                                """)

                # Loss curve and Scatter plot now moved into the scale=2 column
                with gr.Row():
                    with gr.Column(scale=2):  # Increased scale for plots
                        gr.Markdown("### 📉 损失曲线")
                        with gr.Row():
                            epoch_range = gr.Slider(
                                minimum=0,
                                maximum=1,  # Will be updated dynamically
                                value=[0, 1],
                                step=1,
                                label="选择Epoch范围",
                                interactive=True,
                                show_label=True,
                                visible=True,
                                scale=1,
                                elem_id="hidden_epoch_slider"
                            )
                        loss_plot = gr.Plot(
                            value=create_loss_plot([], [], 0),
                            show_label=False
                        )
                    with gr.Column(scale=2):
                        gr.Markdown("### 🎯 预测vs真实值")
                        scatter_plot = gr.Plot(
                            value=create_scatter_plot(None, None),
                            show_label=False
                        )

        # Event handling
        start_btn.click(
            fn=start_training,
            inputs=[seed, valid_ratio, test_ratio, n_epochs, batch_size, learning_rate,
                    early_stop, hidden_size, num_layers, dropout, final_layers_1_dim],
            outputs=result_text
        )

        stop_btn.click(
            fn=stop_training,
            outputs=result_text
        )

        pause_btn.click(
            fn=pause_training,
            outputs=result_text
        )

    def update_all_components():
        global training_state

        status = get_current_status()
        current_epoch_val = "0"
        best_epoch_val = "-"
        current_loss_val = "-"

        # Initialize with no updates, or default plots
        loss_plot_fig = gr.update()
        model_comparison_fig = gr.update()
        attention_bar_fig = gr.update()
        scatter_plot_fig = gr.update()
        epoch_range_update = gr.update()
        metrics_table_df = get_metrics_table()  # Always get metrics table

        # Determine if plots should be updated (only during training or after training completion for final state)
        should_update_plots = training_state['is_training'] or (
                    len(training_state['train_losses']) > 0 and not training_state['is_training'])

        if training_state['train_losses']:
            current_epoch_val = f"{training_state['current_epoch']}"
            best_epoch_val = f"第{training_state['best_epoch'] + 1}轮"
            current_loss_val = f"{training_state['valid_losses'][-1]:.4f}" if training_state['valid_losses'] else "-"

            max_epochs = len(training_state['train_losses'])
            if max_epochs > 0:
                epoch_range_update = gr.update(
                    minimum=0,
                    maximum=max_epochs - 1,  # Slider max index
                    value=[0, max_epochs - 1] if not training_state['is_training'] else [0, training_state[
                        'current_epoch'] - 1],  # Set full range if training done
                    interactive=True,
                    visible=True
                )

            if should_update_plots:
                # Update with current range, or full range if training done
                current_epoch_for_plot = training_state['current_epoch'] - 1  # Adjust for 0-based indexing for plotting
                if not training_state['is_training'] and max_epochs > 0:
                    loss_plot_fig = update_loss_plot([0, max_epochs - 1])
                elif training_state['is_training'] and current_epoch_for_plot >= 0:
                    loss_plot_fig = update_loss_plot([0, current_epoch_for_plot])
                else:
                    loss_plot_fig = gr.update()  # Keep current plot if no new data

                model_comparison_fig = update_model_comparison()
                attention_bar_fig = update_attention_bar()
                scatter_plot_fig = update_scatter_plot()
            else:  # If not training and no losses yet, ensure plots show 'No data' state
                # These should only be called if there's no data AND not training
                if not training_state['train_losses'] and not training_state['is_training']:
                    loss_plot_fig = create_loss_plot([], [], 0)
                    model_comparison_fig = create_model_comparison_plot(None)
                    attention_bar_fig = create_attention_bar(None, None)
                    scatter_plot_fig = create_scatter_plot(None, None)
                else:
                    # If training just finished, keep the last updated plots
                    pass


        else:  # Initial state, no training has started
            loss_plot_fig = create_loss_plot([], [], 0)
            model_comparison_fig = create_model_comparison_plot(None)
            attention_bar_fig = create_attention_bar(None, None)
            scatter_plot_fig = create_scatter_plot(None, None)
            epoch_range_update = gr.update(minimum=0, maximum=1, value=[0, 1], interactive=True, visible=True)

        return [
            status,
            current_epoch_val,
            best_epoch_val,
            current_loss_val,
            model_comparison_fig,
            attention_bar_fig,
            scatter_plot_fig,
            metrics_table_df,  # Use the updated dataframe
            epoch_range_update
        ]

    with demo:
        demo.queue()

        timer = gr.Timer(2)

        def on_timer():
            return update_all_components()

        timer.tick(
            fn=on_timer,
            outputs=[
                status_text, current_epoch_text, best_epoch_text, current_loss_text,
                model_comparison_plot, attention_bar, scatter_plot, metrics_table, epoch_range
            ],
            every=2  # Update every 2 seconds
        )

        demo.load(
            fn=update_all_components,
            outputs=[
                status_text, current_epoch_text, best_epoch_text, current_loss_text,
                model_comparison_plot, attention_bar, scatter_plot, metrics_table, epoch_range
            ]
        )

        epoch_range.change(
            fn=update_loss_plot,
            inputs=epoch_range,
            outputs=loss_plot
        )
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        share=False,
        debug=True,
        show_error=True
    )