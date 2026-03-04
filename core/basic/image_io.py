import numpy as np
from PIL import Image

from core.basic.image_np import ImageNp


class ImageIO:
    def __init__(self):
        pass

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        # 布尔 -> 0/255
        if arr.dtype == bool:
            return (arr.astype(np.uint8) * 255)
        # 已经是 uint8
        if arr.dtype == np.uint8:
            return arr
        # 其他整数或浮点 -> 线性缩放到 0-255
        if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating):
            a = arr.astype(np.float32)
            amin = float(np.nanmin(a))
            amax = float(np.nanmax(a))
            # 若全相等，直接裁剪到 0 或 255 或 cast
            if np.isclose(amax, amin):
                # 如果值接近 1，认为为 0/1，映射为 0/255
                if np.isclose(amax, 1.0):
                    return (a.astype(np.uint8) * 255)
                # 若值为 0，返回全 0
                if np.isclose(amax, 0.0):
                    return np.zeros_like(a, dtype=np.uint8)
                # 其他常数，裁剪到 [0,255]
                return np.clip(a, 0, 255).astype(np.uint8)
            scaled = (a - amin) / (amax - amin) * 255.0
            return np.clip(scaled, 0, 255).astype(np.uint8)
        # 兜底（非常规类型）
        return np.clip(arr.astype(np.float32), 0, 255).astype(np.uint8)

    @staticmethod
    def _split_color_alpha(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """
        输入 uint8 的 numpy 数组，返回 (color, alpha)
        - 若为二维数组 -> color=arr, alpha=None
        - 若为三维数组:
            ch==1 -> color 为二维 (squeeze)，alpha=None
            ch==2 -> color=(H,W) 或 (H,W,1) 转为二维，alpha=(H,W)
            ch==3 -> color=(H,W,3), alpha=None
            ch==4 -> color=(H,W,3), alpha=(H,W)
            其它 -> 尝试取前3个通道为 color，其余忽略
        """
        if arr.ndim == 2:
            return arr, None
        if arr.ndim == 3:
            ch = arr.shape[2]
            if ch == 1:
                return arr[:, :, 0], None
            if ch == 2:
                return arr[:, :, 0], arr[:, :, 1]
            if ch == 3:
                return arr, None
            if ch == 4:
                return arr[:, :, :3], arr[:, :, 3]
            # 其它通道数，尽量保留前三通道作为 color
            if ch > 3:
                return arr[:, :, :3], None
        # 不支持的情况，返回原始作为 color
        return arr, None

    def load_pil_np(self, path: str) -> ImageNp:
        image = Image.open(path)
        mode = image.mode

        # 常见模式处理
        if mode == 'L':  # 8-bit 灰度
            arr = np.array(image)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)

        if mode == 'LA':  # 灰度 + alpha
            arr = np.array(image)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)

        if mode == 'RGB':
            arr = np.array(image)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)

        if mode == 'RGBA':
            arr = np.array(image)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)

        if mode == 'P':  # palette，可能含透明度
            if 'transparency' in image.info:
                conv = image.convert('RGBA')
            else:
                conv = image.convert('RGB')
            arr = np.array(conv)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)

        if mode == '1':  # 1-bit -> 转为 L，保证 0/255
            conv = image.convert('L')
            arr = np.array(conv)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)

        if mode == 'CMYK':
            conv = image.convert('RGB')
            arr = np.array(conv)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)

        if mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
            arr = np.array(image)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)

        # 其它未知模式：优先尝试保留透明信息（转为 RGBA），失败则转 RGB
        try:
            conv = image.convert('RGBA')
            arr = np.array(conv)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)
        except Exception:
            conv = image.convert('RGB')
            arr = np.array(conv)
            arr = self._to_uint8(arr)
            color, alpha = self._split_color_alpha(arr)
            return ImageNp(color, alpha)

    def save_np_pil(self, image_np: ImageNp, path: str):
        """
        将 ImageNp 保存为文件：
        """
        color = image_np.color
        alpha = getattr(image_np, "alpha", None)

        c = self._to_uint8(color)
        if c.ndim == 3 and c.shape[2] == 1:
            c = c[:, :, 0]
        if alpha is not None:
            a = self._to_uint8(alpha)
            if a.ndim == 3 and a.shape[2] == 1:
                a = a[:, :, 0]
            h, w = (c.shape[0], c.shape[1])
            if a.shape[0] != h or a.shape[1] != w:
                a = np.resize(a, (h, w))
            if c.ndim == 2:
                arr = np.stack([c, a], axis=2)  # (H,W,2) -> 'LA'
            elif c.ndim == 3 and c.shape[2] >= 3:
                rgb = c[:, :, :3]
                arr = np.dstack([rgb, a])  # (H,W,4) -> 'RGBA'
            else:
                base = Image.fromarray(c)
                base.putalpha(Image.fromarray(a))
                base.save(path)
                return
            img = Image.fromarray(arr)
            img.save(path)
            return
        if c.ndim == 2:
            img = Image.fromarray(c)  # 'L'
        elif c.ndim == 3 and c.shape[2] >= 3:
            img = Image.fromarray(c[:, :, :3])  # 'RGB'
        else:
            img = Image.fromarray(self._to_uint8(c))
        img.save(path)
