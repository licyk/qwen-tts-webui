import datetime
import string
import random


def generate_datetime_string() -> str:
    """生成日期字符串

    Args:
        str:
            日期字符串
    """
    return datetime.datetime.now().strftime(r"%Y%m%d_%H%M%S")


def generate_random_string(
    length: int | None = 8,
    chars: str | None = None,
    include_uppercase: bool | None = True,
    include_lowercase: bool | None = True,
    include_digits: bool | None = True,
    include_special: bool | None = False,
) -> str:
    """
    生成随机字符串

    Args:
        length (int | None):
            字符串长度, 默认为 8
        chars (str | None):
            自定义字符集, 如果提供则忽略其他参数
        include_uppercase (bool | None):
            是否包含大写字母
        include_lowercase (bool | None):
            是否包含小写字母
        include_digits (bool | None):
            是否包含数字
        include_special (bool | None):
            是否包含特殊字符

    Returns:
        str:
            生成的随机字符串
    """
    if chars is not None:
        char_pool = chars
    else:
        char_pool = ""
        if include_uppercase:
            char_pool += string.ascii_uppercase
        if include_lowercase:
            char_pool += string.ascii_lowercase
        if include_digits:
            char_pool += string.digits
        if include_special:
            char_pool += "!@#$%^&*"

    if not char_pool:
        raise ValueError("字符池不能为空")

    return "".join(random.choice(char_pool) for _ in range(length))


def generate_filename(
    include_date: bool | None = True,
    random_str_len: int | None = 8,
) -> str:
    """生成文件名

    Args:
        include_date (bool | None):
            是否包含日期
        random_str_len (int | None):
            随机字符串的长度

    Returns:
        str:
            文件名字符串
    """
    random_str = generate_random_string(random_str_len)
    if include_date:
        return generate_datetime_string() + f"_{random_str}"

    return random_str
