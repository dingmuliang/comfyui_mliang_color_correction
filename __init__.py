import torch


def _clamp_image(image):
    return image.clamp(0.0, 1.0)


def _rgb_to_hsv(rgb):
    r, g, b = rgb.unbind(dim=-1)
    maxc = torch.max(rgb, dim=-1).values
    minc = torch.min(rgb, dim=-1).values
    delta = maxc - minc

    h = torch.zeros_like(maxc)
    safe_delta = torch.where(delta == 0, torch.ones_like(delta), delta)

    r_is_max = maxc == r
    g_is_max = maxc == g
    b_is_max = maxc == b

    h = torch.where(r_is_max, ((g - b) / safe_delta) % 6.0, h)
    h = torch.where(g_is_max, ((b - r) / safe_delta) + 2.0, h)
    h = torch.where(b_is_max, ((r - g) / safe_delta) + 4.0, h)
    h = torch.where(delta == 0, torch.zeros_like(h), h / 6.0)

    s = torch.where(maxc == 0, torch.zeros_like(maxc), delta / maxc)
    v = maxc
    return torch.stack((h, s, v), dim=-1)


def _hsv_to_rgb(hsv):
    h, s, v = hsv.unbind(dim=-1)
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6).to(torch.int64)
    f = h6 - torch.floor(h6)

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    r = torch.where((i_mod == 0) | (i_mod == 5), v, torch.where((i_mod == 1) | (i_mod == 4), q, p))
    g = torch.where((i_mod == 0) | (i_mod == 3), t, torch.where((i_mod == 1) | (i_mod == 2), v, p))
    b = torch.where((i_mod == 1) | (i_mod == 2), p, torch.where((i_mod == 3) | (i_mod == 4), v, t))
    return torch.stack((r, g, b), dim=-1)


def _luminance(image):
    weights = image.new_tensor([0.2126, 0.7152, 0.0722])
    return torch.sum(image[..., :3] * weights, dim=-1, keepdim=True)


class MLiangColorCorrection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "shadows": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "highlights": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "correct"
    CATEGORY = "image/color"

    def correct(
        self,
        image,
        exposure,
        brightness,
        contrast,
        saturation,
        gamma,
        temperature,
        tint,
        hue_shift,
        shadows,
        highlights,
        strength,
    ):
        original = image
        rgb = image[..., :3].clone()

        rgb = rgb * (2.0 ** exposure)
        rgb = rgb + brightness
        rgb = (rgb - 0.5) * contrast + 0.5

        if temperature != 0.0 or tint != 0.0:
            red = 1.0 + max(temperature, 0.0) * 0.25 - max(-temperature, 0.0) * 0.12
            blue = 1.0 + max(-temperature, 0.0) * 0.25 - max(temperature, 0.0) * 0.12
            green = 1.0 + tint * 0.18
            magenta = 1.0 - tint * 0.09
            rgb = rgb * rgb.new_tensor([red * magenta, green, blue * magenta])

        if saturation != 1.0:
            lum = _luminance(rgb)
            rgb = lum + (rgb - lum) * saturation

        if hue_shift != 0.0:
            hsv = _rgb_to_hsv(_clamp_image(rgb))
            hsv[..., 0] = (hsv[..., 0] + hue_shift / 360.0) % 1.0
            rgb = _hsv_to_rgb(hsv)

        if shadows != 0.0 or highlights != 0.0:
            lum = _luminance(_clamp_image(rgb))
            shadow_mask = (1.0 - lum).pow(2.0)
            highlight_mask = lum.pow(2.0)
            rgb = rgb + shadow_mask * shadows * 0.5
            rgb = rgb + highlight_mask * highlights * 0.5

        if gamma != 1.0:
            rgb = torch.pow(_clamp_image(rgb), 1.0 / gamma)

        corrected = _clamp_image(rgb)
        if image.shape[-1] > 3:
            corrected = torch.cat((corrected, image[..., 3:]), dim=-1)

        if strength < 1.0:
            corrected = original + (corrected - original) * strength

        return (_clamp_image(corrected),)


class MLiangWhiteBalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "green": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "blue": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "normalize_luma": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "balance"
    CATEGORY = "image/color"

    def balance(self, image, red, green, blue, normalize_luma, strength):
        original = image
        rgb = image[..., :3].clone()
        before = _luminance(rgb).mean()
        rgb = rgb * rgb.new_tensor([red, green, blue])
        if normalize_luma:
            after = _luminance(rgb).mean().clamp_min(1e-6)
            rgb = rgb * (before / after)
        corrected = _clamp_image(rgb)
        if image.shape[-1] > 3:
            corrected = torch.cat((corrected, image[..., 3:]), dim=-1)
        if strength < 1.0:
            corrected = original + (corrected - original) * strength
        return (_clamp_image(corrected),)


class CodexColorCorrection(MLiangColorCorrection):
    DEPRECATED = True


class CodexWhiteBalance(MLiangWhiteBalance):
    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "MLiangColorCorrection": MLiangColorCorrection,
    "MLiangWhiteBalance": MLiangWhiteBalance,
    "CodexColorCorrection": CodexColorCorrection,
    "CodexWhiteBalance": CodexWhiteBalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MLiangColorCorrection": "MLiang Color Correction",
    "MLiangWhiteBalance": "MLiang White Balance",
    "CodexColorCorrection": "MLiang Color Correction (Legacy)",
    "CodexWhiteBalance": "MLiang White Balance (Legacy)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
