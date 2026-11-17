"""Compact T2 spinal-cord segmentation adapter.

Adapted from SCT deepseg_/sc.py (Polytechnique Montreal, 2022), LGPL-3.0.
Reference source/license: vendor/spinalcord_reference/. Changes: SPINE.md.
Uses the released CNN weights; centerline tracking replaces SCT's external OptiC.
"""
import base64
from functools import lru_cache
import gzip
import hashlib
import io
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
from PIL import Image
from scipy import ndimage

from .config import MODEL_DIR
from .errors import InputError, ModelUnavailable

SPINE_FILES = {
    "t2_ctr.onnx": (472371, "60773fc31f48c7655c5a75f5936a854407dd51b9b8c196bd850c7ff8d3a9f1a0"),
    "t2_sc.onnx": (7707615, "4654c9798d93e2bdf7edf037ca58ec7d48a64daba55b4ff203426d15c769d055"),
}


def spine_available():
    return all((MODEL_DIR/"spine"/name).is_file() and (MODEL_DIR/"spine"/name).stat().st_size == size
               for name, (size, _) in SPINE_FILES.items())


@lru_cache(maxsize=2)
def _session(name, mtime):
    import onnxruntime as ort
    path = MODEL_DIR/"spine"/name
    if hashlib.sha256(path.read_bytes()).hexdigest() != SPINE_FILES[name][1]:
        raise ModelUnavailable("Spinal cord model checksum mismatch. Run scripts/setup_spine.py.")
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _predict(session, patches):
    output = session.run(["predictions"], {"input_1": np.asarray(patches, dtype=np.float32)})[0]
    if not np.isfinite(output).all() or output.min() < -1e-4 or output.max() > 1.0001:
        raise ModelUnavailable("The segmentation model returned invalid voxel scores.")
    return np.clip(output[..., 0], 0, 1)


def _scale(data):
    lo, hi = np.percentile(data, (2, 98))
    if hi <= lo:
        raise InputError("Insufficient image contrast for spinal cord segmentation.")
    return np.asarray(np.clip((data-lo)/(hi-lo), 0, 1)*255, dtype=np.float32)


def _start(center, width, dimension):
    return max(0, min(int(center)-width//2+1, max(0, dimension-width)))


def _largest(mask):
    components, count = ndimage.label(mask)
    if not count:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    return components == sizes.argmax()


def _locate(slice_data, detector, previous=None):
    nx, ny = slice_data.shape
    if previous is not None:
        positions = [(_start(previous[0], 80, nx), _start(previous[1], 80, ny))]
    else:
        positions = [(x, y) for x in range(0, nx, 80) for y in range(0, ny, 80)]
    # The released detector accepts a dynamic batch dimension. Bound memory to 16 patches.
    best, best_size = None, 0
    for offset in range(0, len(positions), 16):
        coordinates = positions[offset:offset+16]
        batch = np.zeros((len(coordinates),80,80,1), np.float32)
        for i,(x,y) in enumerate(coordinates):
            patch = slice_data[x:x+80,y:y+80]
            batch[i,:patch.shape[0],:patch.shape[1],0] = patch
        predictions = _predict(detector, (batch-51.1417)/57.4408)
        for (x,y), prediction in zip(coordinates, predictions):
            prediction = prediction[:min(80,nx-x),:min(80,ny-y)]
            component = _largest(prediction > .5)
            size = int(component.sum())
            if size > best_size:
                center = ndimage.center_of_mass(component)
                best, best_size = (x+center[0], y+center[1]), size
    if best is None and previous is not None:
        return _locate(slice_data, detector)
    return best


def read_spine(path):
    try:
        image = nib.load(path)
        if len(image.shape) != 3 or min(image.shape) < 3 or np.prod(image.shape) > 64_000_000:
            raise InputError("Spinal cord mode requires one 3D T2 NIfTI volume, at most 64 million voxels.")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3,:3])) < 1e-8:
            raise InputError("Invalid NIfTI geometry.")
        data = image.get_fdata(dtype=np.float32)
        if not np.isfinite(data).all() or np.count_nonzero(data) < 1000:
            raise InputError("The NIfTI is empty or contains invalid intensities.")
        return image, data
    except InputError:
        raise
    except Exception as exc:
        raise InputError("Could not read the spinal cord NIfTI.") from exc


def _overlay(data, mask):
    # Select a sagittal plane through the greatest amount of predicted cord.
    x = int(mask.sum(axis=(1,2)).argmax())
    background = np.rot90(data[x]); foreground = np.rot90(mask[x])
    lo, hi = np.percentile(background, (1,99))
    gray = np.uint8(np.clip((background-lo)/max(float(hi-lo),1e-6),0,1)*255)
    rgb = np.repeat(gray[...,None],3,axis=-1)
    rgb[foreground] = (.35*rgb[foreground]+.65*np.array([90,220,180])).astype(np.uint8)
    image = Image.fromarray(rgb); image.thumbnail((600,600))
    stream = io.BytesIO(); image.save(stream,format="PNG")
    return "data:image/png;base64,"+base64.b64encode(stream.getvalue()).decode()


def predict_spine(path):
    if not spine_available():
        raise ModelUnavailable("Spinal cord weights are missing. Run scripts/setup_spine.py.")
    original, _ = read_spine(path)
    # SCT's RPI notation is the opposite of nibabel's axis-direction notation: LAS.
    orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(original.affine),
                                                 nib.orientations.axcodes2ornt(("L","A","S")))
    oriented = original.as_reoriented(orientation)
    zooms = nib.affines.voxel_sizes(oriented.affine)
    target_zooms = np.array([.5,.5,zooms[2]])
    shape = np.maximum(1,np.round(np.array(oriented.shape)*zooms/target_zooms).astype(int))
    if np.prod(shape) > 40_000_000 or shape[2] > 600:
        raise InputError("Crop the volume to the spinal region before segmentation; the resampled field of view is too large.")
    affine = oriented.affine.copy(); affine[:3,:3] *= target_zooms/zooms
    reference = resample_from_to(oriented,(tuple(shape),affine),order=1)
    data = reference.get_fdata(dtype=np.float32)
    normalized = _scale(data)
    detector, segmenter = [_session(name,(MODEL_DIR/"spine"/name).stat().st_mtime_ns) for name in SPINE_FILES]
    centers = [None]*data.shape[2]
    seed = None
    for z in np.unique(np.linspace(0,data.shape[2]-1,min(17,data.shape[2])).round().astype(int))[::-1]:
        center = _locate(normalized[:,:,z],detector)
        if center is not None:
            seed = int(z); centers[seed] = center; break
    if seed is None:
        raise InputError("The model could not locate a spinal cord. Check T2 contrast, coverage and image quality.")
    for direction in (-1,1):
        previous = centers[seed]; missed = 0
        for z in range(seed+direction,data.shape[2] if direction==1 else -1,direction):
            center = _locate(normalized[:,:,z],detector,previous)
            centers[z] = center
            if center is None:
                missed += 1
                if missed >= 10: break
            else:
                previous = center; missed = 0
    crops = np.zeros((64,64,data.shape[2]),np.float32)
    positions = {}
    for z,center in enumerate(centers):
        if center is None: continue
        x,y = [_start(c,64,n) for c,n in zip(center,data.shape[:2])]
        tile = data[x:x+64,y:y+64,z]
        crops[:tile.shape[0],:tile.shape[1],z] = tile
        positions[z] = (x,y,tile.shape)
    crops = _scale(crops)
    mask = np.zeros(data.shape,np.uint8)
    indices = sorted(positions)
    for offset in range(0,len(indices),16):
        current = indices[offset:offset+16]
        predictions = _predict(segmenter,np.stack([crops[:,:,z] for z in current])[...,None])
        for z,prediction in zip(current,predictions):
            binary = ndimage.binary_fill_holes(_largest(prediction > .7),structure=np.ones((3,3)))
            x,y,size = positions[z]; mask[x:x+size[0],y:y+size[1],z] = binary[:size[0],:size[1]]
    native = resample_from_to(nib.Nifti1Image(mask.astype(np.float32),affine),original,order=1).get_fdata(dtype=np.float32) > .5
    if not native.any():
        raise InputError("The predicted spinal cord mask is empty; no measurements were returned.")
    header = original.header.copy(); header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(native.astype(np.uint8),original.affine,header)
    output.set_qform(original.get_qform(),int(original.header['qform_code']))
    output.set_sform(original.get_sform(),int(original.header['sform_code']))
    # Preview in canonical orientation; downloadable mask retains the original voxel grid.
    canonical = nib.as_closest_canonical(original)
    mask_canonical = nib.as_closest_canonical(output).get_fdata().astype(bool)
    voxels = int(native.sum())
    return {"model":"SCT DeepSeg T2, compact adapter", "source":"https://github.com/spinalcordtoolbox/deepseg_sc_models",
            "device":"cpu", "probabilities":[], "probability_type":"segmentation", "calibrated":False,
            "preview":_overlay(canonical.get_fdata(dtype=np.float32),mask_canonical),
            "segmentation":{"structure":"Spinal cord", "voxel_count":voxels,
                            "volume_ml":round(voxels*abs(float(np.linalg.det(original.affine[:3,:3])))/1000,3),
                            "mask_download_available":True},
            "warnings":["Anatomical segmentation only: this model does not classify tumors, stenosis, or spinal cord disease.",
                        "Compact centerline adaptation; performance has not been independently validated. Inspect the mask with a qualified clinician."],
            "_mask_nifti_gz":gzip.compress(output.to_bytes())}
