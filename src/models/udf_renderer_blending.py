import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


def extract_fields(bound_min, bound_max, resolution, query_func, device):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    ).to(device)
                    val = (
                        query_func(pts)
                        .reshape(len(xs), len(ys), len(zs))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    u[
                        xi * N : xi * N + len(xs),
                        yi * N : yi * N + len(ys),
                        zi * N : zi * N + len(zs),
                    ] = val
    return u


def extract_gradient_fields(bound_min, bound_max, resolution, query_func, device):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution, 3], dtype=np.float32)

    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = torch.meshgrid(xs, ys, zs)
                pts = torch.cat(
                    [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1
                ).to(device)
                val = (
                    query_func(pts)
                    .reshape(len(xs), len(ys), len(zs), 3)
                    .detach()
                    .cpu()
                    .numpy()
                )
                u[
                    xi * N : xi * N + len(xs),
                    yi * N : yi * N + len(ys),
                    zi * N : zi * N + len(zs),
                ] = val
    return u


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    device = weights.device
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples
        ).to(device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    flag = torch.any(torch.isnan(samples)).cpu().numpy().item()
    if flag:
        print("z_vals", samples[torch.isnan(samples)])
        print("z_samples have nan values")
        pdb.set_trace()
        # raise Exception("z_samples have nan values")

    return samples


class UDFRendererBlending:
    def __init__(
        self,
        nerf,
        udf_network,
        deviation_network,
        beta_network,
        color_network,
        vd_network,
        vi_network,
        n_samples,
        n_importance,
        n_outside,
        up_sample_steps,
        perturb,
        sdf2alpha_type="numerical",  # * numerical is better
        upsampling_type="classical",  # classical is better for DTU
        sparse_scale_factor=25000,
        use_norm_grad_for_cosine=False,
        use_unbias_render=True,
        near_surface=0.05,
        device="cuda",
    ):
        # the networks
        self.nerf = nerf
        self.udf_network = udf_network
        self.deviation_network = deviation_network
        self.beta_network = beta_network  # use to detect zero-level set
        self.vd_network = vd_network
        self.vi_network = vi_network
        self.color_network = color_network

        # sampling setting
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.perturb = perturb
        self.up_sample_steps = up_sample_steps
        self.use_unbias_render = use_unbias_render
        self.sdf2alpha_type = sdf2alpha_type
        self.upsampling_type = upsampling_type
        self.sparse_scale_factor = sparse_scale_factor

        self.use_norm_grad_for_cosine = use_norm_grad_for_cosine

        self.sigmoid = nn.Sigmoid()
        self.near_surface = near_surface
        self.device = device

    def udf2logistic(
        self, udf, inv_s, gamma=20, abs_cos_val=1.0, cos_anneal_ratio=None
    ):
        if cos_anneal_ratio is not None:
            abs_cos_val = (abs_cos_val * 0.5 + 0.5) * (
                1.0 - cos_anneal_ratio
            ) + abs_cos_val * cos_anneal_ratio  # always non-positive

        raw = (
            abs_cos_val
            * inv_s
            * torch.exp(-inv_s * udf)
            / (1 + torch.exp(-inv_s * udf)) ** 2
        )
        raw = raw * gamma
        return raw

    def render_core_outside(
        self,
        rays_o,
        rays_d,
        z_vals,
        sample_dist,
        nerf,
        background_rgb=None,
    ):
        batch_size, n_samples = z_vals.shape

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1
        )

        mid_z_vals = z_vals + dists * 0.5

        pts = (
            rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        )  # batch_size, n_samples, 3

        if self.n_outside > 0:
            dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(
                1.0, 1e10
            )
            pts = torch.cat(
                [pts / dis_to_center, 1.0 / dis_to_center], dim=-1
            )  # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        raw, sampled_edge = nerf(pts, dirs)
        alpha = 1.0 - torch.exp(-F.relu(raw.reshape(batch_size, n_samples)) * dists)
        alpha = alpha.reshape(batch_size, n_samples)
        weights = (
            alpha
            * torch.cumprod(
                torch.cat([torch.ones([batch_size, 1]), 1.0 - alpha + 1e-7], -1), -1
            )[:, :-1]
        )  # n_rays, n_samples
        sampled_edge = sampled_edge.reshape(batch_size, n_samples, 3)
        edge = (weights[:, :, None] * sampled_edge).sum(dim=1)
        if background_rgb is not None:
            edge = edge + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            "edge": edge,
            "sampled_edge": sampled_edge,
            "alpha": alpha,
            "weights": weights,
        }

    def up_sample_unbias(
        self,
        rays_o,
        rays_d,
        z_vals,
        udf,
        sample_dist,
        n_importance,
        inv_s,
        beta,
        gamma,
        debug=False,
    ):
        """
        up sampling strategy similar with NeuS;
        only sample more points at the first possible surface intersection
        """
        batch_size, n_samples = z_vals.shape
        pts = (
            rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        )  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)

        udf = udf.reshape(batch_size, n_samples)

        dists_raw = z_vals[..., 1:] - z_vals[..., :-1]
        dists_raw = torch.cat(
            [
                dists_raw,
                torch.Tensor([sample_dist])
                .to(self.device)
                .expand(dists_raw[..., :1].shape),
            ],
            -1,
        )

        dirs = rays_d[:, None, :].expand(pts.shape)

        prev_udf, next_udf = udf[:, :-1], udf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_udf = (prev_udf + next_udf) * 0.5
        mid_z_vals = (prev_z_vals + next_z_vals) * 0.5

        dists = next_z_vals - prev_z_vals

        # !  directly use udf to approximate cos_val, otherwise, sampling will be biased
        fake_sdf = udf
        ## near the 0levelset, the cos_val approximation should be inaccurate
        prev_sdf, next_sdf = fake_sdf[:, :-1], fake_sdf[:, 1:]

        true_cos = (next_sdf - prev_sdf) / (
            next_z_vals - prev_z_vals + 1e-5
        )  # ! the cos_val will be larger than 1
        cos_val = -1 * torch.abs(true_cos)

        prev_cos_val = torch.cat(
            [torch.zeros([batch_size, 1]).to(self.device), cos_val[:, :-1]], dim=-1
        )
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)

        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        # ! leverage the direction of gradients to alleviate the early zero of vis_prob
        vis_mask = (
            torch.ones_like(true_cos).to(self.device).to(true_cos.dtype)
            * (true_cos < 0.05).float()
        )
        vis_mask = vis_mask.reshape(batch_size, n_samples - 1)
        vis_mask = torch.cat(
            [torch.ones([batch_size, 1]).to(self.device), vis_mask], dim=-1
        )

        # * the probability of occlusion
        raw_occ = self.udf2logistic(udf, beta, 1.0, 1.0).reshape(batch_size, n_samples)
        # # near 0levelset alpha_acc -> 1 others -> 0
        alpha_occ = 1.0 - torch.exp(-F.relu(raw_occ) * gamma * dists_raw)

        # # - use the vis_mask, make upsampling more uniform and centered
        vis_prob = torch.cumprod(
            torch.cat(
                [
                    torch.ones([batch_size, 1]).to(self.device),
                    (1.0 - alpha_occ + vis_mask).clip(0, 1) + 1e-7,
                ],
                -1,
            ),
            -1,
        )[
            :, :-1
        ]  # before udf=0 -> 1; after udf=0 -> 0

        # test direct use of vis_mask
        # vis_prob = 1 - vis_mask

        signs_prob = vis_prob[:, :-1]
        sdf_plus = mid_udf
        sdf_minus = mid_udf * -1
        alpha_plus = self.sdf2alpha(
            sdf_plus, cos_val, dists, inv_s
        )  # batch_size, n_samples
        alpha_minus = self.sdf2alpha(sdf_minus, cos_val, dists, inv_s)
        alpha = alpha_plus * signs_prob + alpha_minus * (1 - signs_prob)
        alpha = alpha.reshape(batch_size, n_samples - 1)

        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [torch.ones([batch_size, 1]).to(self.device), 1.0 - alpha + 1e-7],
                    -1,
                ),
                -1,
            )[:, :-1]
        )
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()

        flag = torch.any(torch.isnan(z_samples)).cpu().numpy().item()
        if flag:
            print("z_vals", z_samples[torch.isnan(z_samples)])
            print("z_vals have nan values")
            pdb.set_trace()
            # raise Exception("gradients have nan values")

        return z_samples

    def cat_z_vals(
        self, rays_o, rays_d, z_vals, new_z_vals, udf, net_gradients=None, last=False
    ):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_udf_output = self.udf_network(pts.reshape(-1, 3))[0]
            new_udf_output = new_udf_output.reshape(batch_size, n_importance, -1)
            new_udf = new_udf_output[:, :, 0]
            udf = torch.cat([udf, new_udf], dim=-1)
            xx = (
                torch.arange(batch_size)[:, None]
                .expand(batch_size, n_samples + n_importance)
                .reshape(-1)
            )
            index = index.reshape(-1)
            udf = udf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, udf

    def sdf2alpha(
        self, sdf, true_cos, dists, inv_s, cos_anneal_ratio=None, udf_eps=None
    ):
        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        if cos_anneal_ratio is not None:
            iter_cos = -(
                F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
                + F.relu(-true_cos) * cos_anneal_ratio
            )  # always non-positive
        else:
            iter_cos = true_cos

        abs_cos_val = iter_cos.abs()

        if udf_eps is not None:
            # ? the abs_cos_val might be inaccurate near udf=0
            mask = sdf.abs() < udf_eps
            abs_cos_val[mask] = 1.0  # {udf < udf_eps} use default abs_cos_val value

        if self.sdf2alpha_type == "numerical":
            # todo: cannot handle near surface cases
            estimated_next_sdf = sdf + iter_cos * dists * 0.5
            estimated_prev_sdf = sdf - iter_cos * dists * 0.5

            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

            p = prev_cdf - next_cdf  # ! this decides the ray shoot direction
            c = prev_cdf

            alpha = (p + 1e-5) / (c + 1e-5)
            alpha = alpha.clip(0.0, 1.0)
        elif self.sdf2alpha_type == "theorical":
            raw = abs_cos_val * inv_s * (1 - self.sigmoid(sdf * inv_s))
            alpha = 1.0 - torch.exp(-F.relu(raw) * dists)

        return alpha

    # add by ml
    def gaussian_1d(self, x, sigma, mu=0.0, amplitude=1.0):
        return amplitude * torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    # def squared_constraint_loss(self, vi, vd):
    #     excess = torch.relu(vi + vd -1)
    #     return torch.mean(excess ** 2)

    def threshold_mask(self, tensor, threshold=0.15, radius=3):
        if radius < 0:
            raise ValueError("半径必须是非负整数")

            # 创建布尔掩码（大于阈值的位置为True）
        bool_mask = tensor > threshold

        # 特殊情况处理：半径为0或没有True元素
        if radius == 0 or not bool_mask.any():
            return torch.ones_like(tensor, dtype=torch.float)

        # 转换为浮点型张量
        float_mask = bool_mask.float()
        original_shape = float_mask.shape
        dim = float_mask.dim()

        # 根据维度调整张量形状
        if dim == 2:
            # 2D: (H, W) -> (1, 1, H, W)
            float_mask = float_mask.unsqueeze(0).unsqueeze(0)
        elif dim == 3:
            # 3D: (C, H, W) -> (1, C, H, W)
            float_mask = float_mask.unsqueeze(0)
        elif dim != 4:
            raise ValueError("输入张量必须是2D、3D或4D")

        # 应用最大池化（模拟膨胀操作）
        kernel_size = 2 * radius + 1
        expanded = F.max_pool2d(
            float_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=radius
        )

        # 生成结果掩码（膨胀区域为0，其余为1）
        result = 1 - expanded

        # 恢复原始形状
        if dim == 2:
            result = result.squeeze(0).squeeze(0)
        elif dim == 3:
            result = result.squeeze(0)

        return result

    def surface_mask(self, tensor, threshold=0.02):
        mask = tensor < threshold
        float_mask = mask.float()
        return float_mask

    def conditional_zero_tensor(self, tensor1, tensor2, threshold=0.1, epsilon=1e-6):
        """
        优化后的版本，便于反向传播计算

        参数:
            tensor1 (torch.Tensor): 第一个输入张量
            tensor2 (torch.Tensor): 第二个输入张量，形状与tensor1相同
            threshold (float): 判断元素是否"接近"的阈值
            epsilon (float): 当元素不接近时，tensor1对应位置将被设置为这个接近0的值

        返回:
            torch.Tensor: 处理后的张量
        """
        # 计算绝对差值（保持梯度）
        diff = (tensor1 - tensor2).abs()

        # 创建接近掩码（不需要梯度）
        with torch.no_grad():
            close_mask = diff <= threshold

        # 使用masked_fill替代torch.where，更高效
        # 注意：这里不能直接使用masked_fill，因为我们需要保留梯度
        # 所以采用乘法方式实现
        result = tensor1 * close_mask.to(tensor1.dtype) + (~close_mask).to(tensor1.dtype) * epsilon

        return result

    def render_core(
        self,
        rays_o,
        rays_d,
        z_vals,
        sample_dist,
        udf_network,
        deviation_network,
        color_network,
        vd_network,
        vi_network,
        beta_network=None,
        cos_anneal_ratio=None,
        background_rgb=None,
        background_alpha=None,
        background_sampled_edge=None,
        flip_saturation=0.0,
    ):
        _, n_samples = z_vals.shape
        batch_size, _ = rays_o.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [
                dists,
                torch.Tensor([sample_dist])
                .to(self.device)
                .expand(dists[..., :1].shape),
            ],
            -1,
        )

        mid_z_vals = z_vals + dists * 0.5

        pts = (
            rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        )  # n_rays, n_samples, 3

        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        udf_nn_output, inputs = udf_network(pts)
        udf = udf_nn_output[:, :1]
        feature_vector = udf_nn_output[:, 1:]

        gradients = udf_network.gradient(pts).reshape(batch_size * n_samples, 3)
        gradients_mag = torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True)
        gradients_norm = gradients / (gradients_mag + 1e-5)  # normalize to unit vector

        # vi = vd_network(-1 * gradients, gradients, feature_vector)
        vi_output = vi_network(feature_vector)
        vi = vi_output[:, :1]
        vi_feature = vi_output[:, 1:]

        vd = vd_network(dirs, gradients, vi_feature)
        sampled_color = color_network(dirs, gradients, feature_vector)

        inv_s = deviation_network(torch.zeros([1, 3]).to(self.device))[:, :1].clip(
            1e-6, 1e6
        )  # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        beta = beta_network.get_beta().clip(1e-6, 1e6)
        gamma = beta_network.get_gamma().clip(1e-6, 1e6)

        vis_prob = None
        alpha_occ = None

        if self.use_unbias_render:
            # ? use gradient w/o normalization
            if self.use_norm_grad_for_cosine:
                true_cos = (dirs * gradients_norm).sum(-1, keepdim=True)  # [N, 1]
            else:
                true_cos = (dirs * gradients).sum(-1, keepdim=True)  # [N, 1]

            with torch.no_grad():
                cos = (dirs * gradients_norm).sum(-1, keepdim=True)  # [N, 1]
                flip_sign = (
                    torch.sign(cos) * -1
                )  # used for visualize the surface normal
                flip_sign[flip_sign == 0] = 1

            # * the probability of occlusion
            raw_occ = self.udf2logistic(udf, beta, 1.0, 1.0).reshape(
                batch_size, n_samples
            )

            # near 0levelset alpha_acc -> 1 others -> 0
            alpha_occ = 1.0 - torch.exp(-F.relu(raw_occ) * gamma * dists)

            # ! consider the direction of gradients to alleviate the early zero of vis_prob
            vis_mask = (
                torch.ones_like(true_cos).to(self.device).to(true_cos.dtype)
                * (true_cos < 0.01).float()
            )
            vis_mask = vis_mask.reshape(batch_size, n_samples)

            # shift one pt
            vis_mask = torch.cat(
                [vis_mask[:, 1:], torch.ones([batch_size, 1]).to(self.device)], dim=-1
            )

            vis_prob = torch.cumprod(
                torch.cat(
                    [
                        torch.ones([batch_size, 1]).to(self.device),
                        ((1.0 - alpha_occ + flip_saturation * vis_mask).clip(0, 1))
                        + 1e-7,
                    ],
                    -1,
                ),
                -1,
            )[
                :, :-1
            ]  # before udf=0 -> 1; after udf=0 -> 0

            # testing directly use the vis_mask
            # vis_prob = 1 - vis_mask

            vis_prob = vis_prob.clip(0, 1)

            alpha_plus = self.sdf2alpha(
                udf,
                -1 * torch.abs(true_cos),
                dists.view(-1, 1),
                inv_s,
                cos_anneal_ratio,
            ).reshape(batch_size, n_samples)
            alpha_minus = self.sdf2alpha(
                -udf,
                -1 * torch.abs(true_cos),
                dists.view(-1, 1),
                inv_s,
                cos_anneal_ratio,
            ).reshape(batch_size, n_samples)

            alpha = alpha_plus * vis_prob + alpha_minus * (1 - vis_prob)

            udf = udf.reshape(batch_size, n_samples)

            alpha = alpha.reshape(batch_size, n_samples)

        else:
            udf = udf.reshape(batch_size, n_samples)
            raw_occ = self.udf2logistic(
                udf,
                beta,
                1.0,
                1.0,
            ).reshape(batch_size, n_samples)
            alpha = 1.0 - torch.exp(-F.relu(raw_occ) * gamma * dists)

        sampled_edge = torch.ones((batch_size, n_samples, 1), device=self.device)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(
            batch_size, n_samples
        )
        # inside_sphere = (pts_norm < 1.0).float().detach()
        # relax_inside_sphere = (pts_norm < 1.2).float().detach()
        inside_sphere = (pts_norm < 2.0).float().detach()
        relax_inside_sphere = (pts_norm < 2.4).float().detach()
        near_surface = (udf < self.near_surface).float().detach()

        # Render with background
        if background_alpha is not None:
            # ! introduce biased depth; since first two or three points are outside of sphere; not use inside_sphere
            # alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)

            sampled_edge = torch.cat(
                [sampled_edge, background_sampled_edge[:, n_samples:]], dim=1
            )

            if sampled_edge_pixel is not None:
                sampled_edge_pixel = (
                    sampled_edge_pixel * inside_sphere[:, :, None]
                    + background_sampled_edge[:, :n_samples]
                    * (1.0 - inside_sphere)[:, :, None]
                )
                sampled_edge_pixel = torch.cat(
                    [sampled_edge_pixel, background_sampled_edge[:, n_samples:]],
                    dim=1,
                )

        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [torch.ones([batch_size, 1]).to(self.device), 1.0 - alpha + 1e-7],
                    -1,
                ),
                -1,
            )[:, :-1]
        )

        weights_sum = weights.sum(dim=-1, keepdim=True)

        # surface_mask = self.surface_mask(udf_nn_output[:, :1])
        # vi = vi * surface_mask
        # vd = vd * surface_mask
        # vd_mask = self.threshold_mask(vd_edge)


        cos_angle = torch.sum(dirs * gradients_norm, dim=1, keepdim=True)
        vd_circle = self.gaussian_1d(cos_angle, sigma=0.2)
        vd_error = F.mse_loss(vd, vd_circle)
        vi = vi * (1 - vd_circle)

        vi_edge = (vi.reshape(batch_size, n_samples, 1) * weights[:, :, None]).sum(dim=1)
        vd_edge = (vd.reshape(batch_size, n_samples, 1) * weights[:, :, None]).sum(dim=1)

        edge = vd_edge + vi_edge
        # edge = (vd.reshape(batch_size, n_samples, 1) * weights[:, :, None]).sum(dim=1)
        color = (sampled_color.reshape(batch_size, n_samples, 3) * weights[:,:, None]).sum(dim=1)
        depth = (mid_z_vals * weights[:, :n_samples]).sum(dim=1, keepdim=True)
        if background_rgb is not None:  # Fixed background, usually black
            edge = edge + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error_ = (
            torch.linalg.norm(
                gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1
            )
            - 1.0
        ) ** 2
        gradient_error = (relax_inside_sphere * gradient_error_).sum() / (
            relax_inside_sphere.sum() + 1e-5
        )

        # calculate the eikonal loss near zero level set
        gradient_error_near_surface = (near_surface * gradient_error_).sum() / (
            near_surface.sum() + 1e-5
        )

        gradients = gradients.reshape(batch_size, n_samples, 3)

        # gradients = gradients / (
        #         torch.linalg.norm(gradients, ord=2, dim=-1, keepdim=True) + 1e-5)  # normalize to unit vector

        if torch.any(torch.isnan(gradient_error)).cpu().numpy().item():
            pdb.set_trace()

        if vis_prob is not None:
            # gradients_flip = gradients * vis_prob[:, :, None] + gradients * (1 - vis_prob[:, :, None]) * -1
            gradients_flip = flip_sign.reshape([batch_size, n_samples, 1]) * gradients
        else:
            gradients_flip = gradients

        # geo regularization, encourages the UDF have clear surfaces
        sparse_error = (
            torch.exp(-self.sparse_scale_factor * udf).sum(dim=1, keepdim=False).mean()
        )

        # smooth loss
        near_surface_idx = near_surface.reshape(-1).bool()
        surface_gradients = gradients.reshape(-1, 3)[near_surface_idx]
        surface_gradients = F.normalize(surface_gradients, dim=-1)

        return {
            "udf": udf,
            "edge": edge,
            "color":color,
            "weights": weights,
            "s_val": 1.0 / inv_s,
            "beta": 1.0 / beta,
            "gamma": gamma,
            "depth": depth,
            "gradient_error": gradient_error,
            "gradient_error_near_surface": gradient_error_near_surface,
            "normals": (gradients_flip * weights[:, :n_samples, None]).sum(
                dim=1, keepdim=False
            ),
            "gradients": gradients,
            "gradients_flip": gradients_flip,
            "inside_sphere": inside_sphere,
            "gradient_mag": gradients_mag.reshape(batch_size, n_samples),
            # "true_cos": true_cos.reshape(batch_size, n_samples),
            # "vis_prob": vis_prob.reshape(batch_size, n_samples)
            # if vis_prob is not None
            # else None,
            "alpha": alpha[:, :n_samples],
            "mid_z_vals": mid_z_vals,
            "dists": dists,
            "sparse_error": sparse_error,

            "vi_edge": vi_edge,
            "vd_edge": vd_edge,
            "vd_error": vd_error,
        }

    def render(
        self,
        rays_o,
        rays_d,
        near,
        far,
        depth_scale,
        cos_anneal_ratio=None,
        perturb_overwrite=-1,
        background_rgb=None,
        flip_saturation=0,
        # * blending params
        color_maps=None,
        pose=None,
        fx=None,
        fy=None,
        img_index=None,
        rays_uv=None,
    ):
        batch_size = len(rays_o)

        if not isinstance(near, torch.Tensor):
            near = torch.Tensor([near]).view(1, 1).to(self.device)
            far = torch.Tensor([far]).view(1, 1).to(self.device)

        sample_dist = ((far - near) / self.n_samples).mean().item()
        z_vals = torch.linspace(0.0, 1.0, self.n_samples).to(self.device)

        z_vals = near + (far - near) * z_vals[None, :]
        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(
                1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside
            )

        n_samples = self.n_samples
        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).to(self.device)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = 0.5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                # stratified samples in those intervals
                t_rand = torch.rand(z_vals_outside.shape).to(self.device)
                z_vals_outside = lower + (upper - lower) * t_rand

        if self.n_outside > 0:
            z_vals_outside = (
                far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples
            )

        background_alpha = None
        background_sampled_edge = None
        background_edge = torch.zeros([1, 3])

        # Up sample
        if self.n_importance > 0:
            if self.upsampling_type == "classical":
                z_vals = self.importance_sample(rays_o, rays_d, z_vals, sample_dist)
            elif self.upsampling_type == "mix":
                z_vals = self.importance_sample_mix(rays_o, rays_d, z_vals, sample_dist)

            n_samples = self.n_samples + self.n_importance

        # background model
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(
                rays_o,
                rays_d,
                z_vals_feed,
                sample_dist,
                self.nerf,
                background_rgb=background_rgb,
            )

            background_sampled_edge = ret_outside["sampled_edge"]
            background_alpha = ret_outside["alpha"]

        # ----------------------------------- fine --------------------------------------------
        # render core
        ret_fine = self.render_core(
            rays_o,
            rays_d,
            z_vals,
            sample_dist,
            self.udf_network,
            self.deviation_network,
            self.color_network,
            self.vd_network,
            self.vi_network,
            beta_network=self.beta_network,
            cos_anneal_ratio=cos_anneal_ratio,
            background_rgb=background_rgb,
            background_alpha=background_alpha,
            background_sampled_edge=background_sampled_edge,
            flip_saturation=flip_saturation,
        )

        return {
            "udf": ret_fine["udf"],
            "edge": ret_fine["edge"],
            "color": ret_fine["color"], # add by ml
            "weight_sum": ret_fine["weights"][:, :n_samples].sum(dim=-1, keepdim=True),
            "weight_sum_fg_bg": ret_fine["weights"][:, :].sum(dim=-1, keepdim=True),
            "depth": ret_fine["depth"] * depth_scale,
            "variance": ret_fine["s_val"],
            "beta": ret_fine["beta"],
            "gamma": ret_fine["gamma"],
            "normals": ret_fine["normals"],
            "gradients": ret_fine["gradients"],
            "gradients_flip": ret_fine["gradients_flip"],
            "weights": ret_fine["weights"],
            "gradient_error": ret_fine["gradient_error"],
            "gradient_error_near_surface": ret_fine["gradient_error_near_surface"],
            "inside_sphere": ret_fine["inside_sphere"],
            "gradient_mag": ret_fine["gradient_mag"],
            "mid_z_vals": ret_fine["mid_z_vals"],
            "dists": ret_fine["dists"],
            "vi_edge": ret_fine["vi_edge"],
            "vd_edge": ret_fine["vd_edge"],
            "vd_error": ret_fine["vd_error"],
        }

    @torch.no_grad()
    def importance_sample(self, rays_o, rays_d, z_vals, sample_dist):
        batch_size = rays_o.shape[0]

        # up_sample = self.up_sample_unbias
        if self.use_unbias_render:
            up_sample = self.up_sample_unbias
        else:
            up_sample = self.up_sample_no_occ_aware

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

        udf_nn_output = self.udf_network(pts.reshape(-1, 3))[0]
        udf_nn_output = udf_nn_output.reshape(batch_size, self.n_samples, -1)
        udf = udf_nn_output[:, :, 0]

        for i in range(self.up_sample_steps):
            new_z_vals = up_sample(
                rays_o,
                rays_d,
                z_vals,
                udf,
                sample_dist,
                self.n_importance // self.up_sample_steps,
                64 * 2**i,
                # ! important; use larger beta **(i+1); otherwise sampling will be biased
                64 * 2 ** (i + 1),
                # ! important; use much larger beta **(i+1); otherwise sampling will be biased
                gamma=np.clip(20 * 2 ** (self.up_sample_steps - i), 20, 320),
            )
            z_vals, udf = self.cat_z_vals(
                rays_o,
                rays_d,
                z_vals,
                new_z_vals,
                udf,
                last=(i + 1 == self.up_sample_steps),
            )

        return z_vals

    @torch.no_grad()
    def importance_sample_mix(self, rays_o, rays_d, z_vals, sample_dist):
        """
        This sampling can make optimization avoid bad initialization of early stage
        make optimization more robust
        Parameters
        ----------
        rays_o :
        rays_d :
        z_vals :
        sample_dist :

        Returns
        -------

        """
        batch_size = rays_o.shape[0]

        base_z_vals = z_vals

        # Up sample

        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]

        udf_nn_output = self.udf_network(pts.reshape(-1, 3))[0]
        udf_nn_output = udf_nn_output.reshape(batch_size, self.n_samples, -1)
        udf = udf_nn_output[:, :, 0]
        base_udf = udf

        beta = self.beta_network.get_beta().clip(1e-6, 1e6)
        gamma = self.beta_network.get_gamma().clip(1e-6, 1e6)
        # * not occlussion-aware sample to avoid missing the true surface
        for i in range(self.up_sample_steps):
            new_z_vals = self.up_sample_no_occ_aware(
                rays_o,
                rays_d,
                z_vals,
                udf,
                sample_dist,
                self.n_importance // (self.up_sample_steps + 1),
                64 * 2**i,
                64 * 2 ** (i + 1),
                gamma,
            )
            z_vals, udf = self.cat_z_vals(
                rays_o,
                rays_d,
                z_vals,
                new_z_vals,
                udf,
            )

        for i in range(self.up_sample_steps - 1, self.up_sample_steps):
            new_z_vals = self.up_sample_unbias(
                rays_o,
                rays_d,
                z_vals,
                udf,
                sample_dist,
                self.n_importance // (self.up_sample_steps + 1),
                64 * 2**i,
                # ! important; use larger beta **(i+1); otherwise sampling will be biased
                64 * 2 ** (i + 1),
                # ! important; use much larger beta **(i+1); otherwise sampling will be biased
                gamma=20 if i < 4 else 10,
            )
            z_vals, udf = self.cat_z_vals(
                rays_o,
                rays_d,
                z_vals,
                new_z_vals,
                udf,
                last=(i + 1 == self.up_sample_steps),
            )

        return z_vals

    def up_sample_no_occ_aware(
        self,
        rays_o,
        rays_d,
        z_vals,
        udf,
        sample_dist,
        n_importance,
        inv_s,
        beta,
        gamma,
    ):
        """
        Different with NeuS, here we sample more points at all possible surfaces where udf is close to 0;
        Since unlike that SDF has clear sign changes, UDF sampling may miss the true surfaces
        """
        batch_size, n_samples = z_vals.shape
        pts = (
            rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        )  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)

        udf = udf.reshape(batch_size, n_samples)

        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat(
            [
                dists,
                torch.Tensor([sample_dist])
                .to(self.device)
                .expand(dists[..., :1].shape),
            ],
            -1,
        )

        dirs = rays_d[:, None, :].expand(pts.shape)

        # * the probability of occlusion
        raw_occ = self.udf2logistic(udf, beta, 1.0, 1.0)
        # near 0levelset alpha_acc -> 1 others -> 0
        alpha_occ = 1.0 - torch.exp(
            -F.relu(raw_occ.reshape(batch_size, n_samples)) * gamma * dists
        )

        z_samples = sample_pdf(
            z_vals, alpha_occ[:, :-1], n_importance, det=True
        ).detach()

        flag = torch.any(torch.isnan(z_samples)).cpu().numpy().item()
        if flag:
            print("z_vals", z_samples[torch.isnan(z_samples)])
            print("z_vals have nan values")
            pdb.set_trace()

        return z_samples