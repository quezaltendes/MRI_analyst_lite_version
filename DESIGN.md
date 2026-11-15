# MRI Doctor visual redesign

The visual direction is inspired by [artemdzhalilov.com](https://artemdzhalilov.com/): a near-black canvas, oversized regular-weight Inter typography, grayscale imagery, thin horizontal rules, generous space, numbered editorial sections, and restrained motion.

The application remains local, with its existing FastAPI routes, knee workflow, BrainIAC MCI and IDH workflows, upload validation, inference, and JSON export. The region cards now move to the analysis section when selected. The layout was reviewed at 1280 px and 568 px. Section reveals, hover transitions, and hero animation respect reduced-motion preferences.

Inter is served locally from `mri_doctor/static/assets/InterVariable.woff2`; its upstream license is included alongside the font as `Inter-LICENSE.txt`. No font or illustration request is sent to a third party at runtime.

## Previous hero asset (no longer displayed)

Generated with the built-in ImageGen tool, not the fallback CLI. Project file: `mri_doctor/static/assets/brain-sculpture.png`. The image is an artistic anatomy illustration, not an MRI, patient image, model output, or diagnostic visualization. The reference site's portrait and personal branding were not reused.

Final generation prompt:

> Use case: stylized-concept. Asset type: original monochrome website hero illustration for an MRI research application. Create a single sculptural human brain, isolated and floating against a completely near-black (#050505) background. Grayscale only. Photorealistic matte graphite / silver-gray anatomy sculpture, beautifully intricate cerebral folds, viewed from a front three-quarter angle, centered, brain fills about 75% of square frame, no stand, no skull or face, a little brainstem visible. Large soft directional studio light from upper left creates white highlights and deep black sculptural shadows. Quiet, precise, editorial, museum-quality, subtle fine film grain. Strong contrast, extremely high detail. Keep all of the brain in frame with generous near-black negative space around the perimeter so the image blends into a black webpage. No labels, no text, no UI, no symbols, no neon, no glow, no color. This is an artistic anatomy illustration, not a diagnostic MRI scan.

The screenshot `preview.png` records the redesigned page. Model validation details remain in `README.md` and `validation.json`.

## Current hero slideshow

The hero now displays the three MRI reference images supplied by the user, copied unchanged into `mri_doctor/static/assets/brain-mri-axial.png`, `brain-mri-sagittal.png`, and `brain-mri-coronal.png`. Images crossfade in a 21-second loop, changing every seven seconds with an approximately 1.7-second overlap. CSS `object-fit: contain` preserves each complete frame. Reduced-motion mode displays the first image statically. These are decorative reference images, not uploaded study results or inference outputs. The earlier generated sculpture is retained as an unused asset.
