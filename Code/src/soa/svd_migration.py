import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from .config import SOAConfig

def compute_global_basis(bert_model: BertModel, k_rank: int):
    """
    Computes the global basis matrix U of shape [hidden_size, k_rank].
    Uses PCA/SVD on the concatenated columns of all Q, K, V weight matrices.
    
    Args:
        bert_model: Instance of BertModel or BertForSequenceClassification
        k_rank: Target rank for compression
        
    Returns:
        basis: Tensor of shape [hidden_size, k_rank]
    """
    print(f"🔄 Extracting QKV weights from {len(bert_model.encoder.layer)} layers...")
    
    q_weights = []
    k_weights = []
    v_weights = []
    
    # جمع‌آوری وزن‌ها
    for layer in bert_model.encoder.layer:
        # دریافت وزن‌ها و تبدیل به CPU برای اطمینان
        q_w = layer.attention.self.query.weight.detach().cpu()
        k_w = layer.attention.self.key.weight.detach().cpu()
        v_w = layer.attention.self.value.weight.detach().cpu()
        
        q_weights.append(q_w)
        k_weights.append(k_w)
        v_weights.append(v_w)
    
    # چسباندن وزن‌ها: هر کدام شکل [768, 768] دارند.
    # ما می‌خواهیم همه ستون‌ها را کنار هم بگذاریم تا یک ماتریس بزرگ بسازیم.
    # شکل نهایی: [768, 3 * num_layers * 768]
    W_combined = torch.cat([torch.cat(q_weights, dim=1), 
                            torch.cat(k_weights, dim=1), 
                            torch.cat(v_weights, dim=1)], dim=1)
    
    # انجام SVD
    # U: [H, H], S: [min(H, N)], Vh: [N, H]
    # ما بردارهای ویژه سمت چپ (U) را می‌خواهیم که نشان‌دهنده فضای ورودی/خروجی مشترک هستند.
    U, S, Vh = torch.linalg.svd(W_combined, full_matrices=False)
    
    # انتخاب top-k بردارهای ویژه (ستون‌های اول ماتریس U)
    basis = U[:, :k_rank]
    
    # بررسی نهایی شکل
    if basis.shape != (W_combined.shape[0], k_rank):
        raise RuntimeError(f"Basis shape mismatch: Expected [{W_combined.shape[0]}, {k_rank}], got {basis.shape}")
        
    print(f"✅ Basis computed with final shape: {basis.shape}")
    return basis


# def migrate_bert_to_soa(bert_model: BertModel, soa_model, basis):
#     """
#     Migrates weights from a trained BERT model to an SOA model using the shared basis.
#     """
#     print(f"🔍 Debugging shapes...")
    
#     # 1. استخراج ابعاد از مدل SOA
#     if hasattr(soa_model, 'cfg'):
#         hidden_size = soa_model.cfg.hidden_size
#         k_rank = soa_model.cfg.k_rank
#     else:
#         hidden_size = soa_model.hidden_size
#         k_rank = soa_model.k_rank
        
#     print(f"  SOA hidden_size: {hidden_size}, k_rank: {k_rank}")
#     print(f"  Basis shape: {basis.shape}")

#     # 2. تطبیق ابعاد Basis
#     expected_shape = (hidden_size, k_rank)
#     actual_shape = basis.shape
    
#     if actual_shape[0] != hidden_size:
#         raise RuntimeError(f"Hidden size mismatch: Model expects {hidden_size}, got {actual_shape[0]}")
    
#     if actual_shape[1] != k_rank:
#         print(f"⚠️ Rank adjustment needed...")
#         if actual_shape[1] > k_rank:
#             basis = basis[:, :k_rank]
#         else:
#             new_basis = torch.zeros(expected_shape, dtype=basis.dtype)
#             new_basis[:, :actual_shape[1]] = basis
#             basis = new_basis
            
#     print(f"  Final basis shape: {basis.shape}")

#     # 3. کپی Embeddings
#     print("   Copying embeddings...")
#     try:
#         bert_embeddings = bert_model.bert.embeddings
#     except AttributeError:
#         bert_embeddings = bert_model.embeddings
        
#     soa_model.word_emb.weight.data.copy_(bert_embeddings.word_embeddings.weight.data)
#     soa_model.pos_emb.weight.data.copy_(bert_embeddings.position_embeddings.weight.data)
#     soa_model.seg_emb.weight.data.copy_(bert_embeddings.token_type_embeddings.weight.data)
#     soa_model.ln.weight.data.copy_(bert_embeddings.LayerNorm.weight.data)
#     soa_model.ln.bias.data.copy_(bert_embeddings.LayerNorm.bias.data)

#     # 4. کپی Basis اصلی
#     print("   Copying shared basis...")
#     soa_model.basis.data.copy_(basis)

#     # 5. Pooler & Classifier
#     print("   Checking pooler...")
#     bert_pooler = None
#     if hasattr(bert_model, 'pooler') and bert_model.pooler is not None:
#         bert_pooler = bert_model.pooler
#     elif hasattr(bert_model, 'bert') and hasattr(bert_model.bert, 'pooler') and bert_model.bert.pooler is not None:
#         bert_pooler = bert_model.bert.pooler
        
#     if bert_pooler is not None and hasattr(soa_model, 'pooler'):
#         soa_model.pooler.weight.data.copy_(bert_pooler.dense.weight.data)
#         soa_model.pooler.bias.data.copy_(bert_pooler.dense.bias.data)
#         print("   ✅ Pooler copied.")
#     else:
#         print("   ⚠️ Pooler skipped.")

#     if hasattr(soa_model, 'cls'):
#         print("   ⚠️ Skipping 'cls' layer (needs training).")

#     # 6. Encoder Layers
#     print("   Migrating encoder layers...")
#     U = basis # Shape: [H, R]
    
#     for i, (soa_layer, bert_layer) in enumerate(zip(soa_model.layers, bert_model.bert.encoder.layer)):
#         print(f"      Processing layer {i+1}...")
        
#         # --- LayerNorms ---
#         layer_norms = [m for m in bert_layer.modules() if isinstance(m, nn.LayerNorm)]
#         if len(layer_norms) < 2:
#             raise RuntimeError(f"Could not find 2 LayerNorms in layer {i}. Found: {len(layer_norms)}")
            
#         ln_attn = layer_norms[0]
#         ln_ffn = layer_norms[1]

#         soa_layer.ln1.weight.data.copy_(ln_attn.weight.data)
#         soa_layer.ln1.bias.data.copy_(ln_attn.bias.data)
#         soa_layer.ln2.weight.data.copy_(ln_ffn.weight.data)
#         soa_layer.ln2.bias.data.copy_(ln_ffn.bias.data)
        
#         # --- FFN ---
#         soa_layer.ffn[0].weight.data.copy_(bert_layer.intermediate.dense.weight.data)
#         soa_layer.ffn[0].bias.data.copy_(bert_layer.intermediate.dense.bias.data)
#         soa_layer.ffn[2].weight.data.copy_(bert_layer.output.dense.weight.data)
#         soa_layer.ffn[2].bias.data.copy_(bert_layer.output.dense.bias.data)
        
#         # --- Attention Projections (Q, K, V) via A matrices ---
#         # فرمول: W_orig = U @ A  =>  A = U^T @ W_orig
#         W_Q = bert_layer.attention.self.query.weight.data
#         W_K = bert_layer.attention.self.key.weight.data
#         W_V = bert_layer.attention.self.value.weight.data
        
#         A_Q = U.t() @ W_Q
#         A_K = U.t() @ W_K
#         A_V = U.t() @ W_V
        
#         # تلاش برای پیدا کردن و کپی کردن وزن‌های A با نام‌های احتمالی مختلف
#         def copy_matrix_if_exists(layer, tensor, names):
#             for name in names:
#                 if hasattr(layer, name):
#                     attr = getattr(layer, name)
#                     if isinstance(attr, nn.Parameter):
#                         attr.data.copy_(tensor)
#                     elif isinstance(attr, nn.Linear):
#                         attr.weight.data.copy_(tensor)
#                     print(f"      ✅ Copied to {name}")
#                     return True
#             return False

#         q_names = ['A_q', 'proj_q', 'q_proj', 'query_A']
#         k_names = ['A_k', 'proj_k', 'k_proj', 'key_A']
#         v_names = ['A_v', 'proj_v', 'v_proj', 'value_A']
        
#         copy_matrix_if_exists(soa_layer, A_Q, q_names)
#         copy_matrix_if_exists(soa_layer, A_K, k_names)
#         copy_matrix_if_exists(soa_layer, A_V, v_names)

#     print("✅ Migration completed successfully!")
#     return soa_model


import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from .config import SOAConfig

def compute_global_basis_from_hidden_states(bert_model: BertModel, k_rank: int, sample_size=64):
    """
    Computes the global basis matrix U by analyzing hidden states from random inputs.
    NO access to weights is required.
    
    Args:
        bert_model: Trained model (BertForSequenceClassification or BertModel)
        k_rank: Target rank
        sample_size: Number of random sentences to process
        
    Returns:
        basis: Tensor of shape [hidden_size, k_rank]
    """
    print(f"🔄 Computing Basis from Hidden States ({sample_size} samples)...")
    
    device = next(bert_model.parameters()).device
    bert_model.eval()
    
    # لیست برای ذخیره تمام حالت‌های پنهان (Hidden States) از تمام لایه‌ها
    all_hidden_states = []
    
    # تولید ورودی‌های تصادفی
    # شکل: [batch_size, seq_len]
    dummy_input_ids = torch.randint(0, 30522, (sample_size, 128)).to(device)
    dummy_attention_mask = torch.ones_like(dummy_input_ids).to(device)
    dummy_token_type_ids = torch.zeros_like(dummy_input_ids).to(device)
    
    with torch.no_grad():
        outputs = bert_model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            token_type_ids=dummy_token_type_ids,
            output_hidden_states=True # مهم: درخواست خروجی لایه‌های پنهان
        )
        
        # دریافت hidden_states (لیستی از تانسورها به تعداد لایه‌ها + embedding)
        hidden_states = outputs.hidden_states
        
        # ما معمولاً لایه‌های آخر (مثلاً از لایه ۶ به بعد) یا همه لایه‌ها را ترکیب می‌کنیم
        # اینجا همه لایه‌ها را کنار هم می‌چسبانیم
        for layer_output in hidden_states[1:]: # حذف embedding اول
            # شکل هر لایه: [batch, seq_len, hidden_size]
            # ما می‌خواهیم همه توکن‌ها را فلت کنیم تا یک ماتریس [N_samples * seq_len, hidden_size] بسازیم
            batch_seq, seq_len, hidden_dim = layer_output.shape
            
            # تبدیل به [N, hidden_size]
            reshaped = layer_output.reshape(-1, hidden_dim)
            all_hidden_states.append(reshaped)
    
    # چسباندن تمام حالت‌ها در یک ماتریس بزرگ X
    # شکل نهایی: [Total_Tokens, Hidden_Size]
    X = torch.cat(all_hidden_states, dim=0)
    
    print(f"   Total tokens analyzed: {X.shape[0]}")
    
    # محاسبه کوواریانس یا مستقیماً SVD روی ماتریس داده‌ها
    # روش سریع‌تر: SVD مستقیم روی X (PCA)
    # چون X شکل [N, D] دارد، SVD آن بردارهای ویژه سمت راست (Vh) را می‌دهد که همان فضاهای اصلی هستند.
    # اما ما بردارهای ویژه سمت چپ (U) را می‌خواهیم که فضای ورودی را نشان دهد؟
    # نه، در اینجا ما می‌خواهیم زیرفضایی را پیدا کنیم که داده‌ها در آن قرار دارند.
    # بنابراین، بردارهای ویژه ستونی ماتریس X^T (یا سطری X) مد نظر هستند.
    
    # انجام SVD: X = U * S * Vh
    # U: [N, N] (بیش از حد بزرگ)، S: [min(N,D)]، Vh: [D, D]
    # ما به دنبال Vh هستیم که ستون‌هایش جهت‌های اصلی فضای Hidden State هستند.
    
    # نکته: اگر N خیلی بزرگ باشد، SVD کامل کند است. بهتر است از ایزوتروپی استفاده کنیم.
    # اما برای سادگی و دقت بالا، از full_matrices=False استفاده می‌کنیم.
    
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        # اگر حافظه پر شد، از روش تقریبی استفاده کن (اختیاری)
        print("   ⚠️ Memory limit reached, using randomized SVD approximation...")
        # اینجا می‌توان از trl.randomized_svd استفاده کرد، اما فعلاً فرض بر وجود حافظه کافی است.
        raise

    # بردارهای ویژه سمت راست (Vh) نشان‌دهنده فضاهای اصلی هستند.
    # شکل Vh: [hidden_size, hidden_size]
    # ما اولین k_rank ستون (سطر در Vh) را برمی‌داریم.
    basis = Vh[:k_rank, :].t() # تبدیل به [hidden_size, k_rank]
    
    # بررسی شکل
    if basis.shape != (bert_model.config.hidden_size, k_rank):
        print(f"⚠️ Warning: Adjusting basis shape from {basis.shape} to expected.")
        # اگر شکل درست نبود، سعی کن اصلاحش کنی
        if basis.shape[0] == k_rank and basis.shape[1] == bert_model.config.hidden_size:
             basis = basis.t()
        else:
             # Pad یا truncate کن
             if basis.shape[0] > bert_model.config.hidden_size:
                 basis = basis[:bert_model.config.hidden_size, :]
             if basis.shape[1] > k_rank:
                 basis = basis[:, :k_rank]
             elif basis.shape[1] < k_rank:
                 new_basis = torch.zeros((bert_model.config.hidden_size, k_rank), dtype=basis.dtype)
                 new_basis[:, :basis.shape[1]] = basis
                 basis = new_basis

    print(f"✅ Basis computed successfully with shape: {basis.shape}")
    return basis


# تابع مهاجرت باید همین‌طور باقی بماند، فقط نام تابع فراخوانی تغییر می‌کند
import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from .config import SOAConfig



import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification
from .config import SOAConfig

def compute_global_basis_from_input_states(bert_model: BertModel, k_rank: int, sample_size=64):
    """
    Computes the global basis matrix U by analyzing INPUT states of each layer.
    This follows your instruction: Apply SVD on the input tokens to find the common subspace.
    
    Args:
        bert_model: Trained model
        k_rank: Target rank
        sample_size: Number of random sentences
        
    Returns:
        basis: Tensor of shape [hidden_size, k_rank]
    """
    print(f"🔄 Extracting INPUT states from {len(bert_model.encoder.layer)} layers...")
    
    device = next(bert_model.parameters()).device
    bert_model.eval()
    
    all_input_states = []
    
    # تولید ورودی تصادفی
    dummy_input_ids = torch.randint(0, 30522, (sample_size, 128)).to(device)
    dummy_attention_mask = torch.ones_like(dummy_input_ids).to(device)
    dummy_token_type_ids = torch.zeros_like(dummy_input_ids).to(device)
    
    with torch.no_grad():
        outputs = bert_model(
            input_ids=dummy_input_ids,
            attention_mask=dummy_attention_mask,
            token_type_ids=dummy_token_type_ids,
            output_hidden_states=True
        )
        
        # outputs.hidden_states شامل embedding + لایه ۱ تا ۱۲ است
        for i, hidden_state in enumerate(outputs.hidden_states):
            # شکل: [batch, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = hidden_state.shape
            
            # تبدیل به [N_tokens, hidden_dim]
            reshaped = hidden_state.reshape(-1, hidden_dim)
            all_input_states.append(reshaped)
    
    # چسباندن تمام ورودی‌های تمام لایه‌ها
    X = torch.cat(all_input_states, dim=0)
    
    print(f"   Total input tokens analyzed: {X.shape[0]}")
    
    # انجام SVD روی ماتریس ورودی‌ها
    # X = U * S * Vh
    # ما بردارهای ویژه سمت راست (Vh) را می‌خواهیم که نشان‌دهنده جهت‌های اصلی فضای ورودی هستند.
    try:
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    except RuntimeError:
        print("⚠️ Memory limit reached.")
        raise

    # بردارهای ویژه سمت راست (Vh) نشان‌دهنده فضاهای اصلی هستند.
    # شکل Vh: [hidden_size, hidden_size]
    # ما اولین k_rank سطرها (بردارهای ویژه) را برمی‌داریم.
    # سپس ترانهاده می‌کنیم تا شکل [hidden_size, k_rank] شود.
    basis = Vh[:k_rank, :].t() 
    
    # بررسی شکل نهایی
    if basis.shape != (bert_model.config.hidden_size, k_rank):
        print(f"⚠️ Warning: Adjusting basis shape from {basis.shape} to expected.")
        if basis.shape[0] == k_rank and basis.shape[1] == bert_model.config.hidden_size:
             basis = basis.t()
        else:
             if basis.shape[0] > bert_model.config.hidden_size:
                 basis = basis[:bert_model.config.hidden_size, :]
             if basis.shape[1] > k_rank:
                 basis = basis[:, :k_rank]
             elif basis.shape[1] < k_rank:
                 new_basis = torch.zeros((bert_model.config.hidden_size, k_rank), dtype=basis.dtype)
                 new_basis[:, :basis.shape[1]] = basis
                 basis = new_basis

    print(f"✅ Basis computed successfully from INPUT states with shape: {basis.shape}")
    return basis


def migrate_bert_to_soa(bert_model: BertModel, soa_model, basis):
    """
    Migrates weights from a trained BERT model to an SOA model using the shared basis.
    """
    print(f"🔍 Debugging shapes...")
    
    if hasattr(soa_model, 'cfg'):
        hidden_size = soa_model.cfg.hidden_size
        k_rank = soa_model.cfg.k_rank
    else:
        hidden_size = soa_model.hidden_size
        k_rank = soa_model.k_rank
        
    print(f"  SOA hidden_size: {hidden_size}, k_rank: {k_rank}")
    print(f"  Basis shape: {basis.shape}")
    expected_shape = (hidden_size, k_rank)
    actual_shape = basis.shape
    
    if actual_shape[0] != hidden_size:
        raise RuntimeError(f"Hidden size mismatch: Model expects {hidden_size}, got {actual_shape[0]}")
    
    if actual_shape[1] != k_rank:
        print(f"⚠️ Rank adjustment needed...")
        if actual_shape[1] > k_rank:
            basis = basis[:, :k_rank]
        else:
            new_basis = torch.zeros(expected_shape, dtype=basis.dtype)
            new_basis[:, :actual_shape[1]] = basis
            basis = new_basis
            
    print(f"  Final basis shape: {basis.shape}")
    
    # کپی Embeddings
    print("   Copying embeddings...")
    try:
        bert_embeddings = bert_model.bert.embeddings
    except AttributeError:
        bert_embeddings = bert_model.embeddings
        
    soa_model.word_emb.weight.data.copy_(bert_embeddings.word_embeddings.weight.data)
    soa_model.pos_emb.weight.data.copy_(bert_embeddings.position_embeddings.weight.data)
    soa_model.seg_emb.weight.data.copy_(bert_embeddings.token_type_embeddings.weight.data)
    soa_model.ln.weight.data.copy_(bert_embeddings.LayerNorm.weight.data)
    soa_model.ln.bias.data.copy_(bert_embeddings.LayerNorm.bias.data)
    
    # کپی Basis
    print("   Copying shared basis...")
    soa_model.basis.data.copy_(basis)
    
    # Pooler & Classifier
    print("   Checking pooler and classifier...")
    bert_pooler = None
    if hasattr(bert_model, 'pooler') and bert_model.pooler is not None:
        bert_pooler = bert_model.pooler
    elif hasattr(bert_model, 'bert') and hasattr(bert_model.bert, 'pooler') and bert_model.bert.pooler is not None:
        bert_pooler = bert_model.bert.pooler
    
    if bert_pooler is not None and hasattr(soa_model, 'pooler'):
        soa_model.pooler.weight.data.copy_(bert_pooler.dense.weight.data)
        soa_model.pooler.bias.data.copy_(bert_pooler.dense.bias.data)
        print("   ✅ Pooler copied.")
    else:
        print("   ⚠️ Pooler skipped.")
        
    if hasattr(bert_model, 'classifier') and bert_model.classifier is not None:
        cls_weight = bert_model.classifier.weight.data
        cls_bias = bert_model.classifier.bias.data
        if cls_weight.shape == soa_model.cls.weight.shape and cls_bias.shape == soa_model.cls.bias.shape:
            soa_model.cls.weight.data.copy_(cls_weight)
            soa_model.cls.bias.data.copy_(cls_bias)
            print(f"   ✅ Classifier weights copied from BERT (Shape: {cls_weight.shape})")
        else:
            print(f"   ⚠️ Shape mismatch for Classifier: BERT {cls_weight.shape} vs SOA {soa_model.cls.weight.shape}")
    else:
        print("   ⚠️ No classifier found in source model.")
    
    # Encoder Layers
    print("   Migrating encoder layers...")
    U = basis  # Shape: [H, R]
    bert_encoder_layers = bert_model.bert.encoder.layer if hasattr(bert_model, 'bert') else bert_model.encoder.layer
    
    for i, (soa_layer, bert_layer) in enumerate(zip(soa_model.layers, bert_encoder_layers)):
        print(f"      Processing layer {i+1}...")
        
        # LayerNorms
        layer_norms = [m for m in bert_layer.modules() if isinstance(m, nn.LayerNorm)]
        if len(layer_norms) < 2:
            raise RuntimeError(f"Could not find 2 LayerNorms in layer {i}. Found: {len(layer_norms)}")
            
        ln_attn = layer_norms[0]
        ln_ffn = layer_norms[1]
        soa_layer.ln1.weight.data.copy_(ln_attn.weight.data)
        soa_layer.ln1.bias.data.copy_(ln_attn.bias.data)
        soa_layer.ln2.weight.data.copy_(ln_ffn.weight.data)
        soa_layer.ln2.bias.data.copy_(ln_ffn.bias.data)
        
        # FFN
        soa_layer.ffn[0].weight.data.copy_(bert_layer.intermediate.dense.weight.data)
        soa_layer.ffn[0].bias.data.copy_(bert_layer.intermediate.dense.bias.data)
        soa_layer.ffn[2].weight.data.copy_(bert_layer.output.dense.weight.data)
        soa_layer.ffn[2].bias.data.copy_(bert_layer.output.dense.bias.data)
        
        # QKV Projection via Basis: A = U^T @ W_orig
        W_Q = bert_layer.attention.self.query.weight.data  # [768, 768]
        W_K = bert_layer.attention.self.key.weight.data    # [768, 768]
        W_V = bert_layer.attention.self.value.weight.data  # [768, 768]
        
        A_Q = U.t() @ W_Q  # [R, H] @ [H, H] → [R, H]
        A_K = U.t() @ W_K  # [R, H]
        A_V = U.t() @ W_V  # [R, H]
        
        print(f"      A matrices shape: {A_Q.shape} (q_proj expects {soa_layer.attn.q_proj.weight.shape})")
        
        # ✅ کپی با ترانهاده کردن: A_Q [R,H] → t() → [H,R] = q_proj.weight.shape
        soa_layer.attn.q_proj.weight.data.copy_(A_Q.t())
        soa_layer.attn.k_proj.weight.data.copy_(A_K.t())
        soa_layer.attn.v_proj.weight.data.copy_(A_V.t())
        
        print(f"      ✅ Layer {i+1} attention projections migrated.")
    
    print("✅ Migration completed successfully!")
    return soa_model