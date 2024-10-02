import matplotlib.pyplot as plt
import seaborn as sns

def plot_cluster_characteristics(cluster_characteristics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    features = ['price_change_5m', 'price_change_15m', 'volume_change', 'volatility']
    
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        means = [char[feature]['mean'] for char in cluster_characteristics.values()]
        stds = [char[feature]['std'] for char in cluster_characteristics.values()]
        ax.bar(range(len(means)), means, yerr=stds, capsize=5)
        ax.set_title(f'{feature} by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Mean Value')
    
    plt.tight_layout()
    plt.savefig('cluster_characteristics.png')
    plt.close()

def plot_pre_event_windows(pre_event_windows):
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    features = ['price_change_5m', 'price_change_15m', 'volume_change', 'volatility']
    
    for i, feature in enumerate(features):
        ax = axes[i // 2, i % 2]
        for window in pre_event_windows:
            ax.plot(window.index, window[feature])
        ax.set_title(f'{feature} Before 5x Increase')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('pre_event_windows.png')
    plt.close()