
from PIL import Image
def resize_image(image):
    """
    Resize the image to fit within the specified max_width and max_height while keeping the aspect ratio.
    If attempt_half is True, it will resize the image to half its original size.
    """
    width, height = image.size
    # Resize the image
    resized_image = image.resize((width//2,height//2), Image.Resampling.LANCZOS)
    return resized_image



def resize_image_maxsize(image, max_width=1024, max_height=768):
    """
    Resize the image to fit within the specified max_width and max_height while keeping the aspect ratio.
    """
    width, height = image.size

    # 计算宽度和高度的缩放比例
    width_ratio = max_width / width
    height_ratio = max_height / height

    # 使用较小的缩放比例来保持宽高比
    scale_ratio = min(width_ratio, height_ratio)

    # 计算新的宽度和高度
    new_width = int(width * scale_ratio)
    new_height = int(height * scale_ratio)



    # 按新的尺寸缩放图像
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return resized_image

