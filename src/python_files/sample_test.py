import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. تشخیص دستگاه (MPS برای اپل یا CPU)
# ---------------------------------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ دستگاه فعال: Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ دستگاه فعال: CPU")

os.makedirs("svd_attention_results", exist_ok=True)

# ---------------------------------------------------------
# 2. بارگذاری مدل (با مدیریت خطای اینترنت)
# ---------------------------------------------------------
base_model_name = "bert-base-uncased"

try:
    print(f"در حال بارگذاری توکنایزر...")
    tokenizer = BertTokenizer.from_pretrained(base_model_name)
    
    print(f"در حال بارگذاری کانفیگ...")
    config = BertConfig.from_pretrained(base_model_name)
    
    # تنظیمات برای تست سریع (6 لایه برای سرعت بیشتر)
    config.num_hidden_layers = 6 
    config.hidden_size = 768
    config.num_attention_heads = 12
    
    print("مدل آماده است.")
except Exception as e:
    print(f"❌ خطا در بارگذاری مدل: {e}")
    print("لطفاً اتصال اینترنت خود را بررسی کنید.")
    exit(1)

# ---------------------------------------------------------
# 3. لایه SVD اختصاصی (محاسبه بدون مرتب‌سازی دستی)
# ---------------------------------------------------------
class DirectSVDEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_stats = {}

    def forward(self, hidden_states):
        # تبدیل تنسور به نپایی برای استفاده از np.linalg.svd
        # .detach() برای قطع گراف گرادیانت و .cpu() برای انتقال به حافظه CPU ضروری است
        X = hidden_states.detach().cpu().numpy()
        
        batch_size, seq_len, _ = hidden_states.shape
        
        total_tokens = batch_size * seq_len
        X_flat = X.reshape(total_tokens, -1)
        
        # مرکزیت‌زنی داده‌ها (Centering)
        mean_val = np.mean(X_flat, axis=0, keepdims=True)
        X_centered = X_flat - mean_val
        
        try:
            # محاسبه SVD
            # نکته: np.linalg.svd به طور پیش‌فرض مقادیر ویژه را نزولی مرتب می‌کند.
            # ما اینجا هیچ sort دیگری اعمال نمی‌کنیم و دقیقاً همان خروجی را نگه می‌داریم.
            U, s, Vh = np.linalg.svd(X_centered, full_matrices=False)
        except Exception as e:
            return hidden_states, {}

        eigenvalues = s ** 2
        eigenvectors = Vh.T 
        
        # انتخاب k اولین مولفه (بدون مرتب‌سازی مجدد، چون SVD خودش مرتب کرده)
        k = min(50, len(eigenvalues))
        top_eigenvalues = eigenvalues[:k]
        top_eigenvectors = eigenvectors[:, :k]
        
        # بازگرداندن به تنسور پیروچ
        U_torch = torch.tensor(U, dtype=torch.float32, device=device)
        top_eigenvalues_torch = torch.tensor(top_eigenvalues, dtype=torch.float32, device=device)
        top_eigenvectors_torch = torch.tensor(top_eigenvectors, dtype=torch.float32, device=device)
        
        # محاسبه واریانس تجمعی
        cum_var = np.cumsum(top_eigenvalues) / np.sum(top_eigenvalues)
        
        self.layer_stats = {
            'eigenvalues': top_eigenvalues_torch,
            'eigenvectors': top_eigenvectors_torch,
            'cumulative_var': cum_var,
            'top_k': k
        }
        
        # بازسازی داده‌ها با استفاده از k مولفه اول
        s_torch = torch.sqrt(top_eigenvalues_torch)
        weighted_U = U_torch[:, :k] * s_torch
        output_flat = torch.matmul(weighted_U, top_eigenvectors_torch.T)
        
        output = output_flat.reshape(batch_size, seq_len, self.hidden_size)
        
        return output, self.layer_stats

# ---------------------------------------------------------
# 4. مدل هیبریدی
# ---------------------------------------------------------
class HybridBertWithDirectSVDBert(BertModel):
    def __init__(self, config):
        super().__init__(config)
        # نگه داشتن embedings و pooler اصلی
        self.embeddings = self.embeddings 
        self.pooler = self.pooler
        
        # جایگزینی لایه‌های انکودر با لایه‌های SVD اختصاصی
        self.encoder = nn.ModuleList([
            DirectSVDEncoderLayer(config.hidden_size, config.num_attention_heads)
            for _ in range(config.num_hidden_layers)
        ])
        
        self.init_weights()

# ---------------------------------------------------------
# 5. رسم نمودار (اصلاح شده: نمایش بدون سورت دستی)
# ---------------------------------------------------------
def plot_svd_analysis(layer_idx, stats, all_stats, save_dir="svd_attention_results"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # استخراج داده‌ها
    ev = stats['eigenvalues'].cpu().numpy()
    cv = stats['cumulative_var']
    
    # ایجاد ایندکس‌ها دقیقاً بر اساس طول آرایه (ترتیب طبیعی SVD)
    indices = np.arange(len(ev))
    
    # --- نمودار ۱: مقادیر ویژه (Raw Order) ---
    ax1 = axes[0, 0]
    ax1.bar(indices, ev, color='navy', alpha=0.7, edgecolor='black')
    ax1.set_title(f'Layer {layer_idx}: Eigenvalues (Raw SVD Order)')
    ax1.set_xlabel('Component Index (Natural Order)')
    ax1.set_ylabel('Magnitude')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

    # --- نمودار ۲: واریانس تجمعی (Sequential) ---
    ax2 = axes[0, 1]
    ax2.plot(indices, cv, marker='o', color='green', linewidth=2, markersize=4)
    ax2.axhline(y=0.90, color='red', linestyle='--', label='90% Threshold')
    ax2.set_title(f'Layer {layer_idx}: Cumulative Variance (Sequential)')
    ax2.set_xlabel('Component Index')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.legend(loc='lower right')
    ax2.grid(True)

    # --- نمودار ۳: مقایسه لایه قبلی و فعلی ---
    ax3 = axes[1, 0]
    if layer_idx > 0 and len(all_stats) >= 2:
        prev_ev = all_stats[-2]['eigenvalues'].cpu().numpy()
        curr_ev = stats['eigenvalues'].cpu().numpy()
        
        common_len = min(len(prev_ev), len(curr_ev))
        x_common = np.arange(common_len)
        
        ax3.plot(x_common, prev_ev[:common_len], label=f'Prev ({layer_idx-1})', alpha=0.6, linestyle='--')
        ax3.plot(x_common, curr_ev[:common_len], label=f'Current ({layer_idx})', alpha=0.8, linestyle='-')
        ax3.set_title(f'Comparison: Layer {layer_idx-1} vs {layer_idx}')
        ax3.set_xlabel('Component Index')
        ax3.set_ylabel('Eigenvalue Magnitude')
        ax3.legend()
        ax3.grid(True)
    else:
        ax3.axis('off')
        
    # --- نمودار ۴: اهمیت وزن‌ها (Heatmap عمودی) ---
    ax4 = axes[1, 1]
    norm_ev = ev / np.max(ev)
    
    # رسم به صورت یک ستون عمودی برای نمایش ترتیب دقیق
    im = ax4.imshow(norm_ev.reshape(-1, 1), cmap='YlOrRd', aspect='auto', interpolation='nearest')
    ax4.set_title(f'Weight Importance (Normalized Order)')
    ax4.set_ylabel('Component Index')
    # تنظیم تیک‌ها برای نمایش شماره هر مولفه
    ax4.set_yticks(np.arange(len(norm_ev)))
    ax4.set_yticklabels([str(i) for i in range(len(norm_ev))])
    plt.colorbar(im, ax=ax4, label='Normalized Value')
    
    plt.tight_layout()
    filename = os.path.join(save_dir, f"layer_{layer_idx}_svd_raw.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ نمودار لایه {layer_idx} (بدون سورت دستی) ذخیره شد.")

# ---------------------------------------------------------
# 6. اجرای تست
# ---------------------------------------------------------
def run_direct_svd_test():
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # ایجاد مدل و قرار دادن در حالت ارزیابی (بسیار مهم)
    hybrid_model = HybridBertWithDirectSVDBert(config).to(device)
    hybrid_model.eval() # غیرفعال کردن Dropout و سایر لایه‌های آموزشی
    
    all_stats = []
    
    print("\n--- شروع پردازش با مکانیسم SVD (بدون مرتب‌سازی دستی) ---")
    
    # استخراج Embeddings اولیه
    embedding_output = hybrid_model.embeddings(input_ids=inputs['input_ids'])
    hidden_states = embedding_output
    
    for i, layer_module in enumerate(hybrid_model.encoder):
        # پردازش لایه
        output, layer_info = layer_module(hidden_states)
        
        all_stats.append(layer_info)
        plot_svd_analysis(i, layer_info, all_stats)
        
        # اضافه کردن ریزیدوال (Residual Connection)
        hidden_states = output + embedding_output
        
        # چاپ اطلاعات خلاصه
        top_3 = layer_info['eigenvalues'][:3].cpu().numpy()
        var_total = layer_info['cumulative_var'][-1]
        print(f"لایه {i+1}: Top 3 EV = {top_3}, Total Var Explained = {var_total:.4f}")
    
    print("\n✅ پردازش کامل شد.")
    print("📂 تمام نمودارها در پوشه 'svd_attention_results' ذخیره شدند.")

if __name__ == "__main__":
    run_direct_svd_test()