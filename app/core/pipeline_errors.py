class PipelineStageError(Exception):
    """
    Lỗi phát sinh khi một giai đoạn cụ thể trong quy trình xử lý AI gặp sự cố.
    """

    def __init__(
        self,
        *,
        stage: str,
        code: str,
        message: str,
    ):
        self.stage = stage
        self.code = code
        self.message = message

        super().__init__(message)