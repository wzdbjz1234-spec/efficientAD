from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "EfficientAD-main"


@dataclass(frozen=True)
class ModelArtifacts:
    """Paths needed to restore one model produced under ``output/``."""

    model_dir: Path
    teacher: Path
    student: Path
    autoencoder: Path
    norm_params: Path
    source_dir: Path

    @classmethod
    def from_output(
        cls,
        model: str | Path,
        *,
        dataset: str = "mvtec_ad",
        product: str = "my_product",
        source_dir: str | Path = DEFAULT_SOURCE_DIR,
    ) -> "ModelArtifacts":
        source = Path(source_dir).expanduser().resolve()
        requested = Path(model).expanduser()

        if requested.is_dir():
            model_dir = requested.resolve()
            if not (model_dir / "teacher_final.pth").is_file():
                nested = model_dir / "trainings" / dataset / product
                if nested.is_dir():
                    model_dir = nested
        else:
            model_dir = source / "output" / str(model) / "trainings" / dataset / product

        return cls.from_directory(model_dir, source_dir=source)

    @classmethod
    def from_directory(
        cls,
        model_dir: str | Path,
        *,
        source_dir: str | Path = DEFAULT_SOURCE_DIR,
    ) -> "ModelArtifacts":
        directory = Path(model_dir).expanduser().resolve()
        return cls(
            model_dir=directory,
            teacher=directory / "teacher_final.pth",
            student=directory / "student_final.pth",
            autoencoder=directory / "autoencoder_final.pth",
            norm_params=directory / "norm_params.json",
            source_dir=Path(source_dir).expanduser().resolve(),
        )

    def with_overrides(
        self,
        *,
        teacher: str | Path | None = None,
        student: str | Path | None = None,
        autoencoder: str | Path | None = None,
        norm_params: str | Path | None = None,
    ) -> "ModelArtifacts":
        def resolved(value: str | Path | None, current: Path) -> Path:
            return Path(value).expanduser().resolve() if value else current

        return replace(
            self,
            teacher=resolved(teacher, self.teacher),
            student=resolved(student, self.student),
            autoencoder=resolved(autoencoder, self.autoencoder),
            norm_params=resolved(norm_params, self.norm_params),
        )

    def validate(self, *, require_norm: bool = True) -> "ModelArtifacts":
        required = {
            "EfficientAD source directory": self.source_dir,
            "teacher weights": self.teacher,
            "student weights": self.student,
            "autoencoder weights": self.autoencoder,
        }
        if require_norm:
            required["normalization parameters"] = self.norm_params

        missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing model artifacts:\n  " + "\n  ".join(missing))
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_dir": str(self.model_dir),
            "teacher": str(self.teacher),
            "student": str(self.student),
            "autoencoder": str(self.autoencoder),
            "norm_params": str(self.norm_params),
            "source_dir": str(self.source_dir),
        }
