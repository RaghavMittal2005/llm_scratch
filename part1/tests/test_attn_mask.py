import numpy as np
import torch
from multihead import MultiHeadAttention

# Test data: (B=1, T=3, d_model=4)
X = np.array([[[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.4, 0.3, 0.2],
               [0.0, 0.1, 0.0, 0.1]]], dtype=np.float32)

# Weight matrices for QKV projection (d_model=4 -> 3*d_model=12)
# This will be split into 2 heads, each with d_head=2
W_qkv = np.random.randn(4, 12).astype(np.float32) * 0.1

# Output projection weight (d_model=4 -> d_model=4)
W_proj = np.random.randn(4, 4).astype(np.float32) * 0.1


def test_multihead_basic_shape():
    """Test that MultiHeadAttention produces correct output shapes."""
    torch.manual_seed(42)
    x = torch.tensor(X)
    
    # Create attention module with 2 heads
    attn = MultiHeadAttention(d_model=4, n_head=2, dropout=0.0, trace_shapes=False)
    
    # Load fixed weights for reproducibility
    with torch.no_grad():
        attn.qkv.weight.copy_(torch.tensor(W_qkv).t())
        attn.proj.weight.copy_(torch.tensor(W_proj).t())
    
    out, w = attn(x)
    
    # Check output shape
    assert out.shape == (1, 3, 4), f"Expected output shape (1,3,4), got {out.shape}"
    
    # Check attention weights shape (B, n_head, T, T)
    assert w.shape == (1, 2, 3, 3), f"Expected weights shape (1,2,3,3), got {w.shape}"
    
    # Check for finite values
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"
    assert torch.isfinite(w).all(), "Attention weights contain NaN or Inf"
    
    print("✓ Output shape correct:", out.shape)
    print("✓ Attention weights shape correct:", w.shape)


def test_multihead_causal_mask():
    """Test that causal masking works correctly."""
    torch.manual_seed(42)
    x = torch.tensor(X)
    
    attn = MultiHeadAttention(d_model=4, n_head=2, dropout=0.0, trace_shapes=False)
    
    with torch.no_grad():
        attn.qkv.weight.copy_(torch.tensor(W_qkv).t())
        attn.proj.weight.copy_(torch.tensor(W_proj).t())
    
    out, w = attn(x)
    
    # Check causal masking for each head
    for h in range(2):
        head_weights = w[0, h]  # (T, T)
        
        # Upper triangle should be zero (no attending to future)
        for i in range(3):
            for j in range(i + 1, 3):
                assert head_weights[i, j].item() == 0.0, \
                    f"Head {h}: position ({i},{j}) should be 0, got {head_weights[i, j]}"
        
        # Each row should sum to 1.0 (probability distribution)
        row_sums = head_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(3), atol=1e-5), \
            f"Head {h}: row sums should be 1.0, got {row_sums}"
    
    print("✓ Causal mask applied correctly")
    print("✓ Attention weights sum to 1.0 per row")


def test_multihead_attention_pattern():
    """Test that attention weights follow expected causal pattern."""
    torch.manual_seed(42)
    x = torch.tensor(X)
    
    attn = MultiHeadAttention(d_model=4, n_head=2, dropout=0.0, trace_shapes=False)
    
    with torch.no_grad():
        attn.qkv.weight.copy_(torch.tensor(W_qkv).t())
        attn.proj.weight.copy_(torch.tensor(W_proj).t())
    
    out, w = attn(x)
    
    print("\nAttention weights per head:")
    for h in range(2):
        print(f"\nHead {h}:")
        print(w[0, h].detach().numpy())
        
        # First row should attend only to itself
        assert w[0, h, 0, 0].item() == 1.0, "First token should attend only to itself"
        assert w[0, h, 0, 1].item() == 0.0, "First token shouldn't attend to future"
        assert w[0, h, 0, 2].item() == 0.0, "First token shouldn't attend to future"


def test_multihead_different_heads():
    """Test that different heads learn different attention patterns."""
    torch.manual_seed(42)
    x = torch.tensor(X)
    
    attn = MultiHeadAttention(d_model=4, n_head=2, dropout=0.0, trace_shapes=False)
    
    with torch.no_grad():
        attn.qkv.weight.copy_(torch.tensor(W_qkv).t())
        attn.proj.weight.copy_(torch.tensor(W_proj).t())
    
    out, w = attn(x)
    
    # Check that heads produce different attention patterns
    head0 = w[0, 0]
    head1 = w[0, 1]
    
    # Heads should not be identical (with high probability given random weights)
    assert not torch.allclose(head0, head1, atol=1e-3), \
        "Different heads should learn different patterns"
    
    print("✓ Different heads produce different attention patterns")

