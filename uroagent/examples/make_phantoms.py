"""Draw the synthetic CT teaching phantoms used as examples in the imaging tab.

These are computer-drawn pictures, not scans of any person. They imitate the look of an axial
(cross-sectional) pelvic CT closely enough to try the imaging workflow, without any patient data
or copyright concerns. Run: python make_phantoms.py
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SIZE = 512
HU = {"air": 0, "fat": 70, "fluid": 88, "soft": 125, "muscle": 135, "wall": 140, "contrast": 215, "bone": 235, "balloon": 250}


def _base(seed):
    """Body outline, fat, muscles, pelvic bones and bowel gas common to every phantom."""
    rng = np.random.default_rng(seed)
    img = Image.new("L", (SIZE, SIZE), HU["air"])
    d = ImageDraw.Draw(img)
    d.ellipse([46, 96, 466, 430], fill=HU["fat"])                    # body and subcutaneous fat
    d.ellipse([70, 120, 442, 410], fill=HU["muscle"] - 12)             # deeper tissues
    for x0 in (88, 344):                                               # gluteal and pelvic side-wall muscles
        d.ellipse([x0, 230, x0 + 80, 400], fill=HU["muscle"])
    d.ellipse([196, 112, 240, 160], fill=HU["muscle"]); d.ellipse([272, 112, 316, 160], fill=HU["muscle"])   # rectus muscles
    for x0, x1 in ((100, 170), (342, 412)):                            # acetabula / femoral heads
        d.ellipse([x0, 250, x1, 320], fill=HU["bone"]); d.ellipse([x0 + 14, 264, x1 - 14, 306], fill=HU["bone"] - 50)
    d.ellipse([226, 360, 286, 410], fill=HU["bone"])                   # sacrum / coccyx
    d.ellipse([234, 330, 278, 360], fill=HU["air"] + 20)               # rectal gas
    return img, d, rng


def _finish(img, rng, label):
    arr = np.asarray(img.filter(ImageFilter.GaussianBlur(1.6)), dtype=float)
    arr += rng.normal(0, 6, arr.shape)                                 # CT-like noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr)
    ImageDraw.Draw(out).text((10, SIZE - 18), label, fill=200)
    return out


def spc_catheter():
    img, d, rng = _base(1)
    d.ellipse([186, 170, 326, 300], fill=HU["fluid"])                  # urine-filled bladder
    d.ellipse([186, 170, 326, 300], outline=HU["wall"], width=4)       # thin, normal wall
    d.line([(256, 104), (256, 196)], fill=HU["balloon"] - 30, width=6) # suprapubic catheter tract through the abdominal wall
    d.ellipse([238, 194, 274, 228], outline=HU["balloon"], width=5)    # catheter balloon sitting inside the bladder
    return _finish(img, rng, "SYNTHETIC TEACHING PHANTOM - not a patient")


def bladder_mass():
    img, d, rng = _base(2)
    d.ellipse([186, 170, 326, 300], fill=HU["fluid"])
    d.ellipse([186, 170, 326, 300], outline=HU["wall"], width=4)
    d.chord([180, 196, 236, 262], 90, 270, fill=HU["soft"] + 10)      # irregular soft-tissue mass on the left wall
    d.ellipse([196, 206, 230, 246], fill=HU["soft"] + 18)
    d.line([(170, 210), (150, 196)], fill=HU["soft"], width=3)         # stranding into surrounding fat
    return _finish(img, rng, "SYNTHETIC TEACHING PHANTOM - not a patient")


def hydroureter():
    img, d, rng = _base(3)
    d.ellipse([196, 168, 316, 270], fill=HU["fluid"])
    d.ellipse([196, 168, 316, 270], outline=HU["wall"], width=7)       # thick, trabeculated-looking wall
    for x0 in (186, 286):                                              # both ureters wide and fluid-filled behind the bladder
        d.ellipse([x0, 286, x0 + 40, 326], fill=HU["fluid"]); d.ellipse([x0, 286, x0 + 40, 326], outline=HU["wall"] - 10, width=2)
    d.ellipse([200, 300, 212, 312], fill=HU["bone"])                   # small dense stone in the left ureter near the bladder
    return _finish(img, rng, "SYNTHETIC TEACHING PHANTOM - not a patient")


if __name__ == "__main__":
    for name, fn in [("ct_pelvis_spc_catheter.png", spc_catheter), ("ct_pelvis_bladder_mass.png", bladder_mass),
                     ("ct_pelvis_bilateral_hydroureter.png", hydroureter)]:
        fn().save(name)
        print("wrote", name)
