#!/usr/bin/env python3
"""Decode the controlled box experiment; preserve originals and record provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from PIL import Image, ImageOps, ImageStat, ImageDraw, ExifTags


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('data2'))
    p.add_argument('--output',type=Path,default=Path('outputs/data2-input'))
    p.add_argument('--max-size',type=int,default=2000)
    args=p.parse_args()
    if args.output.exists():
        p.error('Output already exists; choose a new directory')
    if args.max_size < 100:
        p.error('max-size must be >= 100')
    expected={f'IMG_{n}.HEIC' for n in range(3257,3267)}
    actual={x.name for x in args.input.glob('*.HEIC')}
    if actual != expected:
        p.error(f'Unexpected source files: missing={expected-actual}, extra={actual-expected}')
    args.output.mkdir(parents=True)
    manifest={'label_basis':'Capture order, assigned at user direction',
              'groups':{},'max_size':args.max_size,'decoder':'heif-convert (libheif)'}
    sheet=Image.new('RGB',(1500,900),'white'); draw=ImageDraw.Draw(sheet)
    with tempfile.TemporaryDirectory() as tmp:
        for i,n in enumerate(range(3257,3267)):
            state='before' if n<3262 else 'after'
            src=args.input/f'IMG_{n}.HEIC'; decoded=Path(tmp)/f'{n}.jpg'
            subprocess.run(['heif-convert','-q','100',str(src),str(decoded)],check=True,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            with Image.open(decoded) as original:
                exif=original.getexif(); details=exif.get_ifd(34665)
                keep=[271,272,274,306]
                metadata={ExifTags.TAGS.get(k,str(k)):str(exif[k]) for k in keep if k in exif}
                for k in [36867,37386,41989,42036,34855,33434]:
                    if k in details: metadata[ExifTags.TAGS.get(k,str(k))]=str(details[k])
                im=ImageOps.exif_transpose(original).convert('RGB')
                original_size=list(im.size)
                if max(ImageStat.Stat(im.resize((128,128))).stddev)<2:
                    raise RuntimeError(f'Blank or nearly constant decoded image: {src}')
                im.thumbnail((args.max_size,args.max_size),Image.Resampling.LANCZOS)
                dest=args.output/'box'/state/'images'/f'IMG_{n}.jpg'
                dest.parent.mkdir(parents=True,exist_ok=True)
                im.save(dest,quality=97,subsampling=0)
                item={'source':src.name,'sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
                      'image':str(dest.relative_to(args.output)),'size':list(im.size),
                      'oriented_original_size':original_size,'exif':metadata,
                      'image_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                      'focal_35mm':float(details.get(41989,0)) or None}
                manifest['groups'].setdefault(state,[]).append(item)
                im.thumbnail((295,410)); x=i%5*300;y=i//5*450
                sheet.paste(im,(x,y+25));draw.text((x+4,y+5),f'{state}: {src.stem}',fill='black')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    sheet.save(args.output/'contact.jpg')
    print(args.output/'manifest.json')

if __name__=='__main__': main()
