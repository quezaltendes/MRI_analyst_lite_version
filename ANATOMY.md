# Automatic region selection

Auto-detect is enabled when the page opens. Upload one DICOM ZIP or one 3D NIfTI volume; the interface selects Knee or Brain and keeps the uploaded file. Manual region buttons override the selection. Spinal cord studies are also recognized for the optional segmentation mode.

The router first reads explicit anatomy terms from DICOM `BodyPartExamined` / `SeriesDescription`, or the NIfTI `descrip` header. Conflicting anatomical tags require manual selection. Filenames and file extensions never determine the region.

When anatomy tags are absent, representative slices are assembled locally into one contact sheet and sent to the already installed local MedGemma worker. This requires the optional setup described in [REPORTING.md](REPORTING.md); it does not download a separate classifier or send images to a service. The first visual request includes loading and verifying model weights and can take considerably longer than metadata routing. If the worker is unavailable, times out, returns an invalid answer, or cannot identify the region, the application asks for manual selection.

Region selection does not establish a diagnosis, verify sequence type or brain preprocessing, or distinguish MCI from IDH. Brain defaults to the MCI upload form; the user must select the required task and confirm preprocessing. IDH still requires two co-registered volumes. Brain DICOM ZIPs can be identified but must be converted to the supported preprocessed NIfTI input before analysis. NIfTI files inside a ZIP must be extracted first.

`POST /api/detect` accepts multipart `file` and returns `region` (`brain`, `knee`, `spine`, `unknown`), `method`, `message`, `local_only`, and `manual_review`. It does not return disease probabilities or a calibrated anatomy confidence. Uploaded routing files and temporary contact sheets are deleted when the worker finishes, including after client disconnection. The same upload limits and single-worker inference queue apply.

The router has not been validated as a clinical anatomy classifier. Always check the selected region; manual override remains available.
