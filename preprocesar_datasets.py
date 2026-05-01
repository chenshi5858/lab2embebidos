import os
import shutil
from pathlib import Path
from PIL import Image, UnidentifiedImageError

def resize_datasets(base_dir: Path, target_size: int = 96):
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    
    # Encontrar todas las carpetas dataset_grupo_X
    for dataset_folder in base_dir.glob("dataset_grupo_*"):
        if not dataset_folder.is_dir() or dataset_folder.name.endswith("_reescalado"):
            continue
            
        out_folder = base_dir / f"{dataset_folder.name}_reescalado"
        print(f"Procesando {dataset_folder.name} -> {out_folder.name}")
        
        for subfolder_name in ["celular", "esp"]:
            subfolder = dataset_folder / subfolder_name
            if not subfolder.exists() or not subfolder.is_dir():
                continue
                
            out_subfolder = out_folder / subfolder_name
            out_subfolder.mkdir(parents=True, exist_ok=True)
            
            # Copiar etiquetas.txt
            etiquetas_src = subfolder / "etiquetas.txt"
            if etiquetas_src.exists():
                shutil.copy2(etiquetas_src, out_subfolder / "etiquetas.txt")
            
            # Recorrer y redimensionar imagenes
            for path in subfolder.glob("*"):
                if path.is_file() and path.suffix.lower() in valid_extensions:
                    try:
                        with Image.open(path) as img:
                            resample = Image.Resampling.LANCZOS if (img.width > target_size or img.height > target_size) else Image.Resampling.BICUBIC
                            if img.size != (target_size, target_size):
                                img = img.resize((target_size, target_size), resample=resample)
                            
                            dst_path = out_subfolder / path.name
                            save_kwargs = {}
                            if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
                                save_kwargs = {"quality": 95, "optimize": True}
                                
                            img.save(dst_path, **save_kwargs)
                    except (UnidentifiedImageError, OSError) as e:
                        print(f"Error procesando {path.name}: {e}")

if __name__ == "__main__":
    # Suponiendo que el script esta en ./lab2embebidos/ y los datasets en ./
    base_directory = Path(__file__).parent.parent
    resize_datasets(base_directory)
    print("Reescalado completado.")
