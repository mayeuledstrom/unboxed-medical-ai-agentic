from dcm_seg_nodules import extract_seg

seg_path = extract_seg("input/PATIENT_FOLDER", output_dir="results")
print(f"SEG saved to: {seg_path}")
