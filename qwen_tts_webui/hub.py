import os

from huggingface_hub import HfApi
from modelscope.hub.api import HubApi


class HubManager:
    """HuggingFace / ModelScope 库管理器

    Attributes:
        hf_api (HfApi | None):
            HuggingFace API 客户端实例, 用于与 HuggingFace 仓库进行交互
        ms_api (HubApi | None):
            ModelScope API 客户端实例, 用于与 ModelScope 仓库进行交互
        hf_token (str | None):
            HuggingFace 认证令牌, 用于访问私有仓库
        ms_token (str | None):
            ModelScope 认证令牌, 用于访问私有仓库
    """

    def __init__(
        self,
        hf_token: str | None = None,
        ms_token: str | None = None,
    ) -> None:
        """HuggingFace / ModelScope 库初始化

        Args:
            hf_token (str | None):
                HuggingFace Token
            ms_token (str | None):
                ModelScope Token
        """
        self.hf_token = None
        self.ms_token = None
        self.hf_api = None
        self.ms_api = None

        self.relogin(
            hf_token=hf_token,
            ms_token=ms_token,
        )

    def relogin(
        self,
        hf_token: str | None = None,
        ms_token: str | None = None,
    ) -> None:
        """重新使用 Token 登陆 HuggingFace / ModelScope 库

        Args:
            hf_token (str | None):
                HuggingFace Token
            ms_token (str | None):
                ModelScope Token
        """
        self.hf_token = hf_token
        self.ms_token = ms_token

        self.hf_api = HfApi(token=hf_token)
        self.ms_api = HubApi()

        if hf_token is not None:
            os.environ["HF_TOKEN"] = hf_token

        if ms_token is not None:
            os.environ["MODELSCOPE_API_TOKEN"] = ms_token
            self.ms_api.login(access_token=ms_token)
            self.ms_token = ms_token

    @staticmethod
    def get_user_and_repo_name(
        repo_id: str,
    ) -> tuple[str, str]:
        """从仓库 ID 中获取仓库所属的用户名和仓库名字

        Args:
            repo_id (str):
                仓库 ID

        Returns:
            (tuple[str, str]):
                用户名和仓库名字
        """
        user, repo = repo_id.split("/")
        return (user, repo)
