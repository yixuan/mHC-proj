import triton
import triton.language as tl

# ============================================================================
# FORWARD KERNEL: Sinkhorn + save log-history for exact backward
# ============================================================================
@triton.jit
def _mhc_sinkhorn_fwd_kernel(
    W_ptr, M_ptr, H_ptr,
    stride_batch, stride_hist_batch,
    N_LANES: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    NN = N_LANES * N_LANES
    mask = offsets < NN

    w_base = W_ptr + pid * stride_batch
    w_flat = tl.load(w_base + offsets, mask=mask, other=-float("inf"),
                     eviction_policy="evict_last").to(tl.float32)

    # clamp (keep same guardrail you had)
    log_w = tl.maximum(w_flat, -1e5)

    hist_base = H_ptr + pid * stride_hist_batch

    for t in tl.static_range(ITERS):
        # --- Row Norm (log-softmax per row) ---
        for i in tl.static_range(N_LANES):
            row_start = i * N_LANES
            row_mask = (offsets >= row_start) & (offsets < row_start + N_LANES) & mask

            row_data = tl.where(row_mask, log_w, -float("inf"))
            row_max = tl.max(row_data, axis=0)
            row_lse = row_max + tl.log(tl.sum(tl.exp(row_data - row_max), axis=0))
            log_w = tl.where(row_mask, log_w - row_lse, log_w)

        # Save state after row phase (needed for exact backward)
        tl.store(hist_base + (t * 2 + 0) * NN + offsets, log_w, mask=mask)

        # --- Col Norm (log-softmax per col) ---
        for j in tl.static_range(N_LANES):
            col_mask = ((offsets % N_LANES) == j) & mask

            col_data = tl.where(col_mask, log_w, -float("inf"))
            col_max = tl.max(col_data, axis=0)
            col_lse = col_max + tl.log(tl.sum(tl.exp(col_data - col_max), axis=0))
            log_w = tl.where(col_mask, log_w - col_lse, log_w)

        # Save state after col phase (needed for exact backward)
        tl.store(hist_base + (t * 2 + 1) * NN + offsets, log_w, mask=mask)

    m_flat = tl.exp(log_w)
    m_base = M_ptr + pid * stride_batch
    tl.store(m_base + offsets, m_flat, mask=mask)


# ============================================================================
# BACKWARD KERNEL: exact unrolled backward through Sinkhorn loop
# ============================================================================
@triton.jit
def _mhc_sinkhorn_bwd_kernel(
    grad_M_ptr, W_ptr, H_ptr, grad_W_ptr,
    stride_batch, stride_hist_batch,
    N_LANES: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_SIZE)
    NN = N_LANES * N_LANES
    mask = offsets < NN

    base = pid * stride_batch
    hist_base = H_ptr + pid * stride_hist_batch

    # log_w after final col phase
    y_final = tl.load(hist_base + (ITERS * 2 - 1) * NN + offsets,
                      mask=mask, other=-float("inf"))
    m_final = tl.exp(y_final)

    grad_m = tl.load(grad_M_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = grad_m * m_final  # d/dlog = d/dm * exp(log)

    # Reverse over iterations (t_rev = ITERS-1 .. 0)
    for t in tl.static_range(ITERS):
        t_rev = ITERS - 1 - t

        # ---- Backprop through col log-softmax (at iter t_rev) ----
        y_col = tl.load(hist_base + (t_rev * 2 + 1) * NN + offsets,
                        mask=mask, other=-float("inf"))
        p_col = tl.exp(y_col)

        col_sum_map = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for j in tl.static_range(N_LANES):
            col_mask = ((offsets % N_LANES) == j) & mask
            s = tl.sum(tl.where(col_mask, grad, 0.0), axis=0)
            col_sum_map = tl.where(col_mask, s, col_sum_map)

        grad = grad - p_col * col_sum_map

        # ---- Backprop through row log-softmax (at iter t_rev) ----
        y_row = tl.load(hist_base + (t_rev * 2 + 0) * NN + offsets,
                        mask=mask, other=-float("inf"))
        p_row = tl.exp(y_row)

        row_sum_map = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for i in tl.static_range(N_LANES):
            row_start = i * N_LANES
            row_mask = (offsets >= row_start) & (offsets < row_start + N_LANES) & mask
            s = tl.sum(tl.where(row_mask, grad, 0.0), axis=0)
            row_sum_map = tl.where(row_mask, s, row_sum_map)

        grad = grad - p_row * row_sum_map

    # Backprop through clamp: log_w = max(w_flat, -1e5)
    w_flat = tl.load(W_ptr + base + offsets, mask=mask, other=-float("inf")).to(tl.float32)
    grad = tl.where((w_flat > -1e5) & mask, grad, 0.0)

    tl.store(grad_W_ptr + base + offsets, grad, mask=mask)
