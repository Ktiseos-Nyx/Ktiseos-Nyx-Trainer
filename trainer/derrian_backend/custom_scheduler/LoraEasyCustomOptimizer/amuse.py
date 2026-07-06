import torch
import torch.optim

from .utils import copy_stochastic_
import logging

# ---------------------------------------------------------------------------
# Newton-Schulz polar decomposition
# ---------------------------------------------------------------------------

@torch.no_grad()
def _zeropower_core(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Pure-tensor Newton-Schulz iteration.  Designed for ``torch.compile``.

    Assumes *X* is already in the desired compute dtype and ``ndim >= 2``.
    The caller is responsible for dtype conversion and assertion checks.
    """
    a, b, c = 3.4445, -4.7750, 2.0315

    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # Fast-path for the common 2D case (avoids dim/keepdim dispatch overhead).
    if X.ndim == 2:
        X = X / (X.norm() + 1e-7)
    else:
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = A @ X
        X = a * X + b * B + c * A @ B

    if transposed:
        X = X.mT
    return X


@torch._dynamo.utils.disable_cache_limit()
@torch.compile(fullgraph=True, mode="default")
def _zeropower_compiled(X: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Compiled wrapper around :func:`_zeropower_core` for GPU execution."""
    return _zeropower_core(X, steps)


@torch.no_grad()
def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.to(dtype=compute_dtype)
    return _zeropower_compiled(X, steps)


# ---------------------------------------------------------------------------
# Muon update helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    nesterov: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute a Muon-style orthogonalized momentum update.

    .. warning::
        When *nesterov* is ``True`` the *grad* tensor is **mutated in-place**
        via ``grad.lerp_(momentum, beta)``.  Callers that still need the
        original gradient values after this call should pass a copy (the
        bf16/fp16 workspace path in :meth:`AMUSE.step` already does this
        via :meth:`_get_fp32_workspace`).
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:
        update = update.view(update.size(0), -1)

    update = zeropower_via_newtonschulz5(update, compute_dtype=compute_dtype)

    update *= 0.2 * max(update.size(0), update.size(1)) ** 0.5
    return update


# ---------------------------------------------------------------------------
# AMUSE optimizer
# ---------------------------------------------------------------------------

class AMUSE(torch.optim.Optimizer):
    """
    AMUSE optimizer.

    State convention:
    - p stores y while training.
    - eval() converts y -> x using the current beta1.
    - train() converts x -> y using the current beta1.
    - state["z"] stores the anchor z.

    Hyperparameters:
    - lr: top-level learning rate applied as default to all parameter groups.
      Overrides per-type defaults (0.02 for Muon, 3e-4 for AdamW) when set.
      A single lr works for both Muon and AdamW paths because Muon's built-in
      scaling (0.2 * max(M,N)^0.5) normalizes the step size.
    - weight_decay: top-level weight decay applied as default to all groups.
    - heuristic_muon: when True and param_groups is a flat list of tensors,
      automatically splits into Muon (ndim >= 2) and AdamW (ndim < 2) groups.
    - beta1: initial y/x interpolation. During warmup beta1 is constant.
    - rho: controls how quickly beta1 approaches 1 after warmup.
      Higher rho pushes beta1 toward 1 faster, so y moves closer to x
      faster. Lower rho keeps y farther from x for longer.
    - r: polynomial power for the z/x averaging weights.

    Parameter groups:
    - When heuristic_muon=True (recommended), AMUSE automatically applies
      Muon to matrix-valued parameters (ndim >= 2) and AdamW to the rest.
    - When heuristic_muon=False, each group must set {"use_muon": True/False}.
    - Matrix hidden-layer weights use Muon momentum and Newton-Schulz
      orthogonalization.
    - Embeddings, scalar parameters, and output head weights use the
      AdamW-style fallback with beta2.
    - Each group may provide lr and weight_decay; Muon groups may also
      provide momentum.
    """
    def __init__(
        self,
        param_groups,
        *,
        lr: float = None,
        weight_decay: float = 0.0,
        heuristic_muon: bool = True,
        beta1: float = 0.9,
        weight_lr_power: float = 2.0,
        warmup_steps: int = 0,
        rho: float = 1.0,
        r: float = 0.0,
        **kwargs,
    ):
        if warmup_steps <= 0:
            raise ValueError("AMUSE requires warmup_steps > 0.")

        # Warn about unrecognized kwargs
        for key in kwargs:
            logging.warning(
                f"Unrecognized optimizer argument '{key}'. It will be ignored."
            )

        # --- Heuristic Muon auto-split ---
        # When heuristic_muon=True and param_groups is a flat list of tensors
        # (not a list of dicts), automatically split into Muon and AdamW groups
        # based on parameter dimensionality.
        if heuristic_muon:
            if (
                isinstance(param_groups, list)
                and len(param_groups) > 0
                and not isinstance(param_groups[0], dict)
            ):
                muon_params = [p for p in param_groups if p.ndim >= 2]
                adamw_params = [p for p in param_groups if p.ndim < 2]
                param_groups = []
                if muon_params:
                    param_groups.append({"params": muon_params, "use_muon": True})
                if adamw_params:
                    param_groups.append({"params": adamw_params, "use_muon": False})
                if not param_groups:
                    param_groups = [{"params": [], "use_muon": False}]
            else:
                logging.warning(
                    "heuristic_muon=True but param_groups is already a list of "
                    "dicts; using explicit group assignments."
                )

        self.beta1_init = float(beta1)
        self.weight_lr_power = weight_lr_power
        self.warmup_steps = int(warmup_steps)
        self.rho = float(rho)
        self.r = r
        self.train_mode = False
        self._srng_buf = None  # Reusable int32 scratch for stochastic rounding

        super().__init__(param_groups, defaults={})
        for group in self.param_groups:
            # Apply top-level lr/weight_decay as defaults (can be overridden
            # per-group by providing lr/weight_decay in the group dict).
            if lr is not None:
                group.setdefault("lr", lr)
            if weight_decay is not None:
                group.setdefault("weight_decay", weight_decay)

            group.setdefault("warmup_steps", self.warmup_steps)
            group.setdefault("k", 0)
            group.setdefault("weight_sum", 0.0)
            group.setdefault("use_muon", False)
            group.setdefault("weight_decay", 0.0)
            group.setdefault("beta1", self.beta1_init)

            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group["params"] = sorted(
                    group["params"], key=lambda x: x.size(), reverse=True
                )
                for p in group["params"]:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("beta2", 0.999)
                group.setdefault("eps", 1e-10)
                for p in group["params"]:
                    state = self.state[p]
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p)

            # Capture base_lr AFTER setdefault("lr") so the default is used
            # when the caller doesn't provide an explicit lr.
            group["base_lr"] = group.get("lr", 0.0)

    def _get_srng_buf(self, like_tensor: torch.Tensor) -> torch.Tensor:
        """Get a reusable int32 scratch buffer for stochastic rounding noise.

        Returns a view of a single shared buffer sized to the largest parameter
        encountered. The buffer is reused across all parameters within a step,
        eliminating per-parameter int32 allocations.
        """
        n = like_tensor.numel()
        if (
            self._srng_buf is None
            or self._srng_buf.device != like_tensor.device
            or self._srng_buf.numel() < n
        ):
            self._srng_buf = torch.empty(
                n, dtype=torch.int32, device=like_tensor.device
            )
        return self._srng_buf[:n].view(like_tensor.shape)

    def _compute_beta1(self, group, t, c_t, warmup_steps):
        if t <= warmup_steps:
            if t == warmup_steps:
                group["c_warmup"] = c_t
            return self.beta1_init

        c_warmup = group.get("c_warmup", 1.0 / warmup_steps)
        S_t = (c_t * (1.0 - c_warmup)) / (c_warmup * (1.0 - c_t))
        return 1.0 - (S_t ** self.rho) * (1.0 - self.beta1_init)

    def _get_z(self, p):
        state = self.state[p]
        z = state.get("z")
        if z is None:
            z = state["z"] = torch.clone(p, memory_format=torch.preserve_format)
        return z

    def _lerp_to_z(self, p, z, weight, state):
        """Lerp p toward z with stochastic rounding for reduced-precision params.

        For bf16/fp16 params, uses cached fp32 workspaces from *state* and
        writes back with stochastic rounding.  For fp32 params, lerps directly.

        :param p: Parameter tensor.
        :param z: Anchor tensor (same shape as *p*).
        :param weight: Lerp interpolation weight.
        :param state: Optimizer state dict for *p* (used for workspace caching).
        """
        if p.dtype in (torch.bfloat16, torch.float16):
            srng = self._get_srng_buf(p)
            p_fp32 = self._get_fp32_workspace(state, "_ws_p", p)
            z_fp32 = self._get_fp32_workspace(state, "_ws_z", z, device=p.device)
            p_fp32.lerp_(end=z_fp32, weight=weight)
            copy_stochastic_(p.data, p_fp32, scratch=srng)
        else:
            z_dev = z.to(device=p.device)
            p.lerp_(end=z_dev, weight=weight)

    def _batch_lerp_to_z(self, group, weight):
        """Batch lerp all params in *group* toward z using ``_foreach``.

        Groups parameters by precision path:

        * **fp32 params** — batched via :func:`torch._foreach_lerp_`.
        * **bf16/fp16 params** — fp32 workspace copies are batched via
          :func:`torch._foreach_lerp_`, then each result is written back
          with stochastic rounding (per-tensor, since
          :func:`copy_stochastic_` is ``@torch.compiler.disable``).

        :param group: Parameter group dict.
        :param weight: Lerp interpolation weight (scalar).
        """
        fp32_pairs = []       # (p, z) for fp32 direct foreach lerp
        rprec_triples = []    # (p, z, state) for reduced-precision path

        for p in group["params"]:
            state = self.state[p]
            if "z" not in state:
                continue
            z = state["z"]
            if p.dtype in (torch.bfloat16, torch.float16):
                rprec_triples.append((p, z, state))
            else:
                fp32_pairs.append((p, z))

        # ---- fp32 path: single batched foreach lerp ----
        if fp32_pairs:
            # Use .data to avoid leaf Variable in-place restriction with
            # torch._foreach_lerp_ (even inside @torch.no_grad).
            p_data_list = [p.data for p, _ in fp32_pairs]
            z_list = [z.to(device=p.device) for p, z in fp32_pairs]
            torch._foreach_lerp_(p_data_list, z_list, weight)

        # ---- reduced-precision path: foreach lerp on fp32 workspaces ----
        if rprec_triples:
            p_fp32_list = []
            z_fp32_list = []
            srng_list = []
            p_list = []

            for p, z, state in rprec_triples:
                srng_list.append(self._get_srng_buf(p))
                p_fp32_list.append(
                    self._get_fp32_workspace(state, "_ws_p", p)
                )
                z_fp32_list.append(
                    self._get_fp32_workspace(state, "_ws_z", z, device=p.device)
                )
                p_list.append(p)

            torch._foreach_lerp_(p_fp32_list, z_fp32_list, weight)

            for p, p_fp32, srng in zip(p_list, p_fp32_list, srng_list):
                copy_stochastic_(p.data, p_fp32, scratch=srng)

    def _get_fp32_workspace(self, state, key, source_tensor, device=None):
        """Get or create a cached fp32 workspace tensor in optimizer state.

        Avoids repeated ``.to(dtype=torch.float32)`` allocations by reusing a
        persistent fp32 buffer stored under ``state[key]``.  The source tensor
        is copied into the workspace on every call so the buffer always reflects
        the latest values.

        :param state: Optimizer state dict for the parameter.
        :param key: State key for the workspace (e.g. ``"_ws_p"``).
        :param source_tensor: The tensor whose values to copy into the workspace.
        :param device: Optional target device.  Defaults to ``source_tensor.device``.
        :returns: A float32 tensor on *device* containing a copy of *source_tensor*.
        """
        dev = device or source_tensor.device
        ws = state.get(key)
        if (
            ws is None
            or ws.device != dev
            or ws.shape != source_tensor.shape
        ):
            ws = state[key] = torch.empty(
                source_tensor.shape, dtype=torch.float32, device=dev
            )
        ws.copy_(source_tensor)
        return ws

    @torch.no_grad()
    def eval(self):
        if self.train_mode:
            for group in self.param_groups:
                beta1 = group.get("beta1", self.beta1_init)
                self._batch_lerp_to_z(group, 1.0 - 1.0 / beta1)
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        if not self.train_mode:
            for group in self.param_groups:
                beta1 = group.get("beta1", self.beta1_init)
                self._batch_lerp_to_z(group, 1.0 - beta1)
        self.train_mode = True

    @torch.no_grad()
    def step(self, closure=None):
        if not self.train_mode:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the optimizer."
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Pre-compute the fused lerp weight shared by all code paths.
        # Replaces the two-step sequence:
        #   p.lerp_(z, ckp1)  then  p.lerp_(z, 1-beta1)
        # with a single equivalent lerp from x_t:
        #   p.lerp_(z, 1 - beta1 * (1 - ckp1))
        # (Derived from the schedule-free interpolation algebra.)

        for group in self.param_groups:
            base_lr = group["base_lr"]
            k = group["k"]
            warmup_steps = group.get("warmup_steps", self.warmup_steps)
            if warmup_steps <= 0:
                raise ValueError("AMUSE requires warmup_steps > 0.")

            t = k + 1
            sched = min(1.0, t / warmup_steps)
            lr = base_lr * sched
            group["lr"] = lr

            # ckp1 is the new averaging weight (c_{t+1} in paper notation).
            weight = (t ** self.r) * (lr ** self.weight_lr_power)
            future_weight_sum = group.get("weight_sum", 0.0) + weight
            ckp1 = weight / future_weight_sum if future_weight_sum > 0 else 1.0
            group["ckp1"] = ckp1
            group["weight_sum"] = future_weight_sum

            # β_t schedule uses the current ckp1 (paper Appendix C.4 exact
            # formula).  Matches the reference implementation which passes
            # the newly computed ckp1 to _compute_beta1.
            beta1 = self._compute_beta1(group, t, ckp1, warmup_steps)
            group["beta1"] = beta1
            wd = group.get("weight_decay", 0.0)

            # Fused lerp weight: replaces two sequential lerps
            #   p.lerp_(z, ckp1) then p.lerp_(z, 1-beta1)
            # with a single lerp from x_t: p.lerp_(z, 1 - beta1*(1-ckp1))
            fused_lerp_w = 1.0 - beta1 * (1.0 - ckp1)

            if group.get("use_muon", False):
                beta_m = group["momentum"]
                for p in group["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    state = self.state[p]

                    z = self._get_z(p)

                    if p.dtype in (torch.bfloat16, torch.float16):
                        srng = self._get_srng_buf(p)
                        # Use cached fp32 workspaces to avoid per-step allocations
                        p_fp32 = self._get_fp32_workspace(state, "_ws_p", p)
                        grad_fp32 = self._get_fp32_workspace(
                            state, "_ws_grad", grad
                        )
                        z_fp32 = self._get_fp32_workspace(
                            state, "_ws_z", z, device=p.device
                        )
                        mom_fp32 = self._get_fp32_workspace(
                            state, "_ws_mom", state["momentum_buffer"],
                            device=p.device,
                        )

                        # y_t -> x_t, then update z, then rebuild y_{t+1}.
                        p_fp32.lerp_(end=z_fp32, weight=1.0 - 1.0 / beta1)
                        # Use fp32 Newton-Schulz for fp16 params (bf16 HW support not assumed)
                        ns_dtype = (
                            torch.float32
                            if p.dtype == torch.float16
                            else torch.bfloat16
                        )
                        update = muon_update(
                            grad_fp32,
                            mom_fp32,
                            beta=beta_m,
                            nesterov=True,
                            compute_dtype=ns_dtype,
                        )
                        if wd != 0.0:
                            z_fp32.mul_(1.0 - lr * wd)
                        z_fp32.add_(
                            update if update.shape == p.shape else update.reshape(p.shape),
                            alpha=-lr,
                        )
                        p_fp32.lerp_(end=z_fp32, weight=fused_lerp_w)

                        copy_stochastic_(p.data, p_fp32, scratch=srng)
                        copy_stochastic_(z, z_fp32, scratch=srng)
                        copy_stochastic_(
                            state["momentum_buffer"], mom_fp32, scratch=srng
                        )
                    else:
                        # y_t -> x_t, then update z, then rebuild y_{t+1}.
                        p.lerp_(end=z, weight=1.0 - 1.0 / beta1)
                        update = muon_update(
                            grad,
                            state["momentum_buffer"],
                            beta=beta_m,
                            nesterov=True,
                        )
                        if wd != 0.0:
                            z.mul_(1.0 - lr * wd)
                        z.add_(
                            update if update.shape == p.shape else update.reshape(p.shape),
                            alpha=-lr,
                        )
                        p.lerp_(end=z, weight=fused_lerp_w)
            else:
                beta2 = group.get("beta2", 0.999)
                eps = group.get("eps", 1e-10)
                bias_correction2 = 1.0 - beta2 ** t
                for p in group["params"]:
                    grad = p.grad
                    if grad is None:
                        continue
                    state = self.state[p]

                    z = self._get_z(p)

                    if p.dtype in (torch.bfloat16, torch.float16):
                        srng = self._get_srng_buf(p)
                        # Use cached fp32 workspaces to avoid per-step allocations
                        p_fp32 = self._get_fp32_workspace(state, "_ws_p", p)
                        grad_fp32 = self._get_fp32_workspace(
                            state, "_ws_grad", grad
                        )
                        z_fp32 = self._get_fp32_workspace(
                            state, "_ws_z", z, device=p.device
                        )
                        v_fp32 = self._get_fp32_workspace(
                            state, "_ws_v", state["exp_avg_sq"],
                            device=p.device,
                        )

                        p_fp32.lerp_(end=z_fp32, weight=1.0 - 1.0 / beta1)
                        v_fp32.mul_(beta2).addcmul_(
                            grad_fp32, grad_fp32, value=1.0 - beta2
                        )
                        denom = v_fp32.div(bias_correction2).sqrt_().add_(eps)
                        update = grad_fp32.div_(denom)
                        if wd != 0.0:
                            update.add_(z_fp32, alpha=wd)
                        z_fp32.add_(update, alpha=-lr)
                        p_fp32.lerp_(end=z_fp32, weight=fused_lerp_w)

                        copy_stochastic_(p.data, p_fp32, scratch=srng)
                        copy_stochastic_(z, z_fp32, scratch=srng)
                        copy_stochastic_(
                            state["exp_avg_sq"], v_fp32, scratch=srng
                        )
                    else:
                        p.lerp_(end=z, weight=1.0 - 1.0 / beta1)
                        v = state["exp_avg_sq"]
                        v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                        denom = v.div(bias_correction2).sqrt_().add_(eps)
                        update = grad.div_(denom)
                        if wd != 0.0:
                            update.add_(z, alpha=wd)
                        z.add_(update, alpha=-lr)
                        p.lerp_(end=z, weight=fused_lerp_w)
            group["k"] = k + 1

        return loss
