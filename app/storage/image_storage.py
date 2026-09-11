from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from uuid import uuid4


class LocalImageStorage:
    """
    Lưu trữ cục bộ bền vững cho các hình ảnh kiểm tra.

    Trả về khóa lưu trữ tương đối thay vì để lộ
    đường dẫn hệ thống tệp tuyệt đối cho API hoặc cơ sở dữ liệu.
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def save(self, source_path: Path) -> str:
        """
        Sao chép hình ảnh vào bộ nhớ lưu trữ kiểm tra cố định.

        Trả về:
            Khóa lưu trữ tương đối, ví dụ:
            inspections/2026/09/abc123.jpg
        """

        source_path = Path(source_path)

        if not source_path.exists():
            raise FileNotFoundError(
                f"Source image does not exist: {source_path}"
            )

        now = datetime.now(timezone.utc)

        relative_dir = Path(
            str(now.year),
            f"{now.month:02d}",
        )

        filename = f"{uuid4().hex}{source_path.suffix.lower()}"

        destination_dir = self.root_dir / relative_dir
        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_path = destination_dir / filename

        copy2(
            source_path,
            destination_path,
        )

        return str(
            Path("inspections")
            / relative_dir
            / filename
        ).replace("\\", "/")

    def resolve(self, storage_key: str) -> Path:
        """
        Resolve a relative storage key to an absolute path.
        """

        relative_path = Path(storage_key)

        if relative_path.is_absolute():
            raise ValueError(
                "Storage key must be relative."
            )

        if ".." in relative_path.parts:
            raise ValueError(
                "Invalid storage key."
            )

        # storage_key:
        # inspections/2026/09/file.jpg
        #
        # root_dir:
        # data/inspections
        #
        # We remove the leading "inspections".
        parts = relative_path.parts

        if not parts or parts[0] != "inspections":
            raise ValueError(
                "Invalid inspection storage key."
            )

        return self.root_dir.joinpath(
            *parts[1:]
        )

    def exists(self, storage_key: str) -> bool:
        return self.resolve(storage_key).exists()