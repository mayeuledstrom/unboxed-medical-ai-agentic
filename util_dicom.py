import requests
import matplotlib.pyplot as plt
import numpy as np
import pydicom
from pathlib import Path
from dcm_seg_nodules import extract_seg
from web_interface import chat_return_image

def get_info_lesions(patientID: str, AccessionNumber: str, out_dir="/home/jovyan/work/dataset"):
    res = []
    serie_valide = get_serie_valide(patientID, AccessionNumber, out_dir)
    img, pxSpacing = get_array(serie_valide)
    lesions = get_lesions(serie_valide, pxSpacing, out_dir)
    for lesion in lesions:
        bbox = get_bounding_box(lesion, pxSpacing)
        res.append({"volume":get_volume(lesion, pxSpacing), "dimensions":(bbox[0]*pxSpacing[0], bbox[1]*pxSpacing[0], bbox[2]*pxSpacing[0])})
    return res

def get_volume(array_lesion, resolution: float):
    return array_lesion.sum() * resolution[0]**3

def get_bounding_box(array, resolution):
    indices = np.where(array)
    
    x_min, x_max = np.min(indices[0]), np.max(indices[0])
    y_min, y_max = np.min(indices[1]), np.max(indices[1])
    z_min, z_max = np.min(indices[2]), np.max(indices[2])
    
    # Dimensions de l'espace contenant tous les True
    x_dim = x_max - x_min + 1
    y_dim = y_max - y_min + 1
    z_dim = z_max - z_min + 1

    return (x_dim, y_dim, z_dim)
    
def get_serie_valide(patientID: str, AccessionNumber: str, out_dir="/home/jovyan/work/dataset") -> str:
    """Retourne le chemin de la série contenant 'torax'"""
    try:
        study_dir = Path(out_dir) / patientID / AccessionNumber
        
        if not study_dir.exists():
            print(f"  Dossier non trouvé : {study_dir}")
            return None
        
        for item in study_dir.iterdir():

            if item.is_dir():
                try:
                    extract_seg(str(item))
                    return str(item)
                except Exception as e:
                    pass
        
        print(f"  Aucune série trouvée contenant 'torax'")
        return None
    
    except Exception as e:
        print(f"  Erreur : {e}")
        return None
        
def show_image(img):
    """Affiche une coupe DICOM avec ses métadonnées principales."""
    plt.imshow(img, cmap="gray", interpolation="bilinear")
    plt.show()

# Takes the path of a serie as input
def get_array(path, the_type=np.float32):
    parent_path = Path(path)
    res = []
    pxSpacing = 0
    for f in parent_path.iterdir():
        if f.is_file():
            ds = pydicom.dcmread(f)
            pxSpacing = ds.get("PixelSpacing")
            the_img = ds.pixel_array.astype(the_type)
            res.append(the_img)
    return (np.array(res), pxSpacing)

def get_lesions(input_serie, pxSpacing, output_folder = "/home/jovyan/work/output"):
    seg_path = extract_seg(input_serie, output_dir=output_folder)
    #print(seg_path[1])
    lesions = np.squeeze(get_array(Path(seg_path[0]).parent, np.bool)[0])
    z_size = get_array(input_serie)[0].shape[0]
    nb_lesions = lesions.shape[0] // z_size
    res = []
    for i in range(nb_lesions):
        res.append(lesions[i*z_size : (i+1)*z_size, :, :])
    return np.array(res)
    
def get_lesion_barycentre(lesion_array: np.ndarray) -> np.ndarray:
    """
    Calcule les coordonnées du barycentre d'une lésion.

    Retourne un np array de dim 1 3 avec les coordonnées du barycentre
    """
    indices = np.where(lesion_array)
    
    # calcul la moyenne des coordonnées pour avoir le barycentre
    z_barycentre = np.mean(indices[0])
    x_barycentre = np.mean(indices[1])
    y_barycentre = np.mean(indices[2])
    
    return np.array([z_barycentre, x_barycentre, y_barycentre])

def get_lesions_position(patientID: str, AccessionNumber: str, etudes_avec_bbox: dict, out_dir="/home/jovyan/work/dataset"):
    """
    Calcule la position relative en pourcentage de chaque lésion par rapport à la bounding box des poumons.
    Retourne un np array de shape (n_lesions, 3) contenant les pourcentages [z%, x%, y%] pour chaque lésion 
    """
    
    serie_valide = get_serie_valide(patientID, AccessionNumber, out_dir)
    if serie_valide is None:
        return None
    
    img, pxSpacing = get_array(serie_valide)
    lesions = get_lesions(serie_valide, pxSpacing)
    
    # Récupère la bounding box des poumons
    bbox = etudes_avec_bbox[patientID][AccessionNumber]['bbox']
    if not bbox:
        print(f"Bounding box non disponible pour {patientID}/{AccessionNumber}")
        return None
    
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    
    bbox_x_size = x_max - x_min
    bbox_y_size = y_max - y_min
    bbox_z_size = z_max - z_min
    
    relative_positions = []
    
    for lesion in lesions:

        barycentre = get_lesion_barycentre(lesion)
        z_barycentre, x_barycentre, y_barycentre = barycentre
        

        z_percent = ((z_barycentre - z_min) / bbox_z_size) * 100
        x_percent = ((x_barycentre - x_min) / bbox_x_size) * 100
        y_percent = ((y_barycentre - y_min) / bbox_y_size) * 100
        
        # Clamp les valeurs entre 0 et 100 au cas où la lésion s'étendrait hors de la bbox
        z_percent = np.clip(z_percent, 0, 100)
        x_percent = np.clip(x_percent, 0, 100)
        y_percent = np.clip(y_percent, 0, 100)
        
        relative_positions.append([z_percent, x_percent, y_percent])
    
    return np.array(relative_positions)


def show_overlay(dicom_img, lesion, slice_idx, color='green', alpha=0.3):
    """
    Affiche le DICOM avec la lésion colorée par-dessus, sans modifier le fond.
    
    Parameters:
    - dicom_img: image DICOM de shape (91, 512, 512)
    - lesion: image booléenne de shape (91, 512, 512) avec True/False
    - slice_idx: index de la coupe à afficher
    - color: couleur de la lésion ('red', 'green', 'blue', 'yellow', 'cyan', 'magenta')
    - alpha: transparence de la lésion (0 à 1)
    
    Returns:
    - result: array RGB de la superposition (512, 512, 3)
    """
    # Normaliser l'image DICOM entre 0 et 1
    dicom_norm = (dicom_img[slice_idx] - dicom_img[slice_idx].min()) / (dicom_img[slice_idx].max() - dicom_img[slice_idx].min())
    
    # Créer une image RGB du DICOM en gris
    dicom_rgb = np.stack([dicom_norm, dicom_norm, dicom_norm], axis=2)
    
    # Créer une image RGB pour la lésion colorée
    lesion_slice = lesion[slice_idx].astype(bool)
    colored_lesion = np.zeros((*lesion_slice.shape, 3))
    
    # Colormap
    color_map = {
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'blue': [0, 0, 1],
        'yellow': [1, 1, 0],
        'cyan': [0, 1, 1],
        'magenta': [1, 0, 1]
    }
    
    color_rgb = color_map.get(color, [0, 1, 0])
    colored_lesion[lesion_slice] = color_rgb
    
    # Mélanger les deux images : fond DICOM + lésion colorée
    result = dicom_rgb.copy()
    mask_3d = np.stack([lesion_slice, lesion_slice, lesion_slice], axis=2)
    result[mask_3d] = (1 - alpha) * dicom_rgb[mask_3d] + alpha * colored_lesion[mask_3d]
    
    return result

    





"""class Client:
    def __init__(self, url, username, password, out_dir="/home/jovyan/work/dataset"):
        self.ORTHANC = url
        self.AUTH = (username, password)
        self.out_dir = out_dir

    def get_studies(self):
        return requests.get(f"{self.ORTHANC}/studies", auth=self.AUTH, timeout=5).json()

    def get_patient_id(self, study_id):
        return 

    def get_path(self, study_id) -> str:
        return Path(self.out_dir) / f"study_{study_id[:8]}.zip"
    
    def download_study(self, study_id: str) -> str:
        #Télécharge une étude complète (.zip) depuis Orthanc.

        dest = self.get_path(study_id)
        if dest.exists():
            file_size = dest.stat().st_size
            print(f"  ℹ️  Fichier existe déjà : {dest} ({file_size/1e6:.1f} Mo)")
            print(f"  ⏭️  Téléchargement ignoré")
            return str(dest)

        print(f"  ⬇️  Téléchargement de l'étude {study_id[:12]}…")
        with requests.get(f"{self.ORTHANC}/studies/{study_id}/archive",
                        auth=self.AUTH, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
                    total += len(chunk)
        print(f"  ✅ Sauvegardé : {dest}  ({total/1e6:.1f} Mo)")
        
        return str(dest)

    def download_and_extract_study(self, study_id: str) -> str:
        #Télécharge et extrait automatiquement une étude

        zip_path = self.download_study(study_id)
        zip_path = Path(zip_path)

        extract_dir = zip_path.parent / zip_path.stem
        self.local_paths[study_id] = extract_dir
        return str(extract_dir)


    def get_series(self, study_id):
        study_dest = self.get_path(study_id)
        return [item for item in path.iterdir() if item.is_dir()]

    def show_dicom(path: str):
        #Affiche une coupe DICOM avec ses métadonnées principales.
        ds  = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), facecolor="#111")
        ax.imshow(img, cmap="gray", interpolation="bilinear")
        ax.set_title(
            f"Patient: {getattr(ds,'PatientID','?')}  |  "
            f"Modality: {getattr(ds,'Modality','?')}  |  "
            f"Slice: {getattr(ds,'InstanceNumber','?')}",
            color="white", fontsize=10, pad=10
        )
        ax.axis("off")
        plt.tight_layout()
        plt.show()

        print(f"  Dimensions   : {img.shape}")
        print(f"  Pixel spacing: {getattr(ds,'PixelSpacing','N/A')}")
        print(f"  Study date   : {getattr(ds,'StudyDate','N/A')}")
        return ds

    def get_study_date_by_accession(accession_number: str):
       
        #Retourne la StudyDate correspondant à un AccessionNumber, None si non trouvé.
       
        studies_ids = requests.get(f"{ORTHANC}/studies", auth=AUTH, timeout=5).json()
    
        for sid in studies_ids:
            info = requests.get(f"{ORTHANC}/studies/{sid}", auth=AUTH, timeout=5).json()
            tags = info.get("MainDicomTags", {})
            
            if tags.get("AccessionNumber") == accession_number:
                return tags.get("StudyDate")
    
        return None  """