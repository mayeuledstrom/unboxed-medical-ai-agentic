from util_dicom import *
from web_interface import *

def frame2D(lesionID: int, patientID: str, AccessionNumber: str, source_folder="/home/jovyan/work/dataset", output_folder="/home/jovyan/work/output"):
    
    # On récupère la série valide
    input_serie = get_serie_valide(patientID, AccessionNumber)
    
    img_base, pxSpacing = get_array(input_serie)
    lesion_array = get_lesions(input_serie, pxSpacing, output_folder=output_folder)
    lesion = lesion_array[lesionID]  # Récupérer la bonne lésion
    
    barycentre = get_lesion_barycentre(lesion)
    barycentre_z = int(barycentre[0])

    result = show_overlay(img_base, lesion, barycentre_z)

    string_result= chat_return_image(result)

    return string_result