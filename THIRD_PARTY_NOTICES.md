# Third-Party Notices

This repository is a fork of FoundationVision/ByteTrack at commit
`d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`, distributed under the MIT License.
The upstream MIT license text is preserved in this repository.

## YOLOX

The detector architecture and training code originate from Megvii YOLOX
(Apache-2.0). The pretrained YOLOX-S COCO checkpoint is downloaded at runtime
from the official Megvii-BaseDetection release and is **not** committed here.
The license of the pretrained checkpoint is governed by its upstream release;
verify the upstream terms before redistributing any checkpoint file.

## FFmpeg

FFmpeg is a separate program that must be installed by the user and placed on
`PATH`. It is not redistributed with this repository. FFmpeg builds carry their
own license terms depending on configuration; consult the FFmpeg project for
details.

## Datasets and media

Sample media used during development carries its own Creative Commons license
and is **not** relicensed under MIT. See `docs/data-manifest.json` for source,
author, license, and download metadata. Video files are not committed to this
repository.

## No affiliation

This project is an independent engineering exercise. It is not affiliated with,
endorsed by, or a product of any commercial vendor, and it makes no claim about
railway control, safety-critical systems, or industrial deployment.
