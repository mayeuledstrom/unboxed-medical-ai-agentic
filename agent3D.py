from util_dicom import *
from web_interface import *

def frame3D(lesionId: int, patientID: str, AccessionNumber: str, src_dir="/home/jovyan/work/dataset", out_dir="/home/jovyan/work/data/"):
    serie_valide = get_serie_valide(patientID, AccessionNumber, src_dir)
    arr, pxSpacing = get_array(serie_valide)
    lesions = get_lesions(serie_valide, pxSpacing, out_dir)
    return chat_return_obj(lesions[lesionId], out_dir)