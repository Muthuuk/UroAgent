import os

from uroagent.imaging import analyse, NoVisionBackend
from uroagent.webapp import IMAGING_EXAMPLES, MODALITIES


def test_sample_files_exist_and_modalities_valid():
    for files, modality, question, consent in IMAGING_EXAMPLES:
        assert modality in MODALITIES
        assert consent is True and question
        for f in files:
            assert os.path.exists(f), f


def test_sample_reports_flag_cancer_and_obstruction():
    out = {}
    for files, modality, question, _ in IMAGING_EXAMPLES:
        if files[0].endswith(".txt"):
            md = analyse(files, modality=modality, question=question, backend=NoVisionBackend(), consent=True)["markdown"].lower()
            out[os.path.basename(files[0])] = md
    assert "pi-rads" in out["report_mri_prostate_pirads4.txt"]
    assert "bladder tumour" in out["report_cystoscopy_spc.txt"]
    assert "bladder stone" not in out["report_cystoscopy_spc.txt"]       # "No stones seen" is a negative
    assert "hydro" in out["report_ct_urogram.txt"]
    assert "ureteric or kidney stone" in out["report_ct_urogram.txt"]
    assert "bladder stone" not in out["report_ct_urogram.txt"]
