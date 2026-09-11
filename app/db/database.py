from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import DATABASE_URL


class Base(DeclarativeBase):
    # Tất cả các bảng/model đều sẽ kế thừa từ class Base => SQLAlchemy nhận diện cấu trúc bảng
    pass


# Khởi tạo đối tượng engine để quản lý kết nối vật lý tới database  
engine = create_engine( 
    DATABASE_URL,
    pool_pre_ping=True, # SQLAlchemy sẽ tự động kết nối lại thay vì báo lỗi
)

SessionLocal = sessionmaker(
    bind=engine,        # Gắn phiên làm việc vào engine đã tạo
    autoflush=False,    # Không tự động đẩy các thay đổi tạm thời xuống database
    autocommit=False,
    # Mặc định không tự lưu thay đổi - phải gọi lệnh db.commit() 
    # Để đảm bảo an toàn dữ liệu 
)


def get_db():
    db = SessionLocal() # Mở 1 phiên làm việc mới với db khi có req gọi tới API

    try:
        yield db        # Cung cấp db đó cho API xử lý logic
    finally:
        db.close()      # Đóng kết nối ngay lập tức để tránh tràn bộ nhớ connection pool