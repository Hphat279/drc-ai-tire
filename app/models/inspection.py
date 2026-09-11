from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class InspectionRun(Base):
    __tablename__ = "inspection_runs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    image_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )
    
    segmentation_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    detection_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )
    
    processing_time_ms: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )

    detections: Mapped[list["InspectionDetection"]] = relationship(
        back_populates="inspection",
        cascade="all, delete-orphan",
    )

    fields: Mapped[list["InspectionField"]] = relationship(
        back_populates="inspection",
        cascade="all, delete-orphan",
    )


class InspectionDetection(Base):
    __tablename__ = "inspection_detections"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    inspection_id: Mapped[int] = mapped_column(
        ForeignKey(
            "inspection_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    class_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    x1: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    y1: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    x2: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    y2: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    inspection: Mapped["InspectionRun"] = relationship(
        back_populates="detections",
    )


class InspectionField(Base):
    __tablename__ = "inspection_fields"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    inspection_id: Mapped[int] = mapped_column(
        ForeignKey(
            "inspection_runs.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    field_name: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
    )

    extraction_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    ocr_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    source_detections: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    inspection: Mapped["InspectionRun"] = relationship(
        back_populates="fields",
    )