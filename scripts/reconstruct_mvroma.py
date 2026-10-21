#!/usr/bin/env python3
"""MV-RoMa matching + custom COLMAP adapter for independent static image groups.

This is not a reproduction of the paper's unpublished SfM post-processing.
Requires the official MV-RoMa checkout, checkpoint, and CUDA inference environment.
"""
import argparse
from argparse import Namespace
import hashlib
import itertools
import inspect
from importlib.metadata import version as package_version
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mvroma_geometry import KeypointIndex, sample_pair, checkpoint_for_model, validate_cache


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()


def source_identity(root):
    """Fingerprint tracked inference sources, including local modifications."""
    names=subprocess.check_output(['git','-C',str(root),'ls-files','-z']).decode().split('\0')
    h=hashlib.sha256()
    for name in sorted(n for n in names if n.endswith(('.py','.toml','.txt'))):
        path=root/name
        h.update(name.encode());h.update(path.read_bytes() if path.exists() else b'<deleted>')
    return {'commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
            'source_sha256':h.hexdigest()}


def infer(args, manifest, cache):
    import torch
    sys.path.insert(0,str(args.mvroma.resolve()))
    from src.build_model import build_our_model
    from src.matchers.uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
    from src.mvroma import ModelConfig
    from src.run_model import run_model_test
    cfg=ModelConfig(); cfg.num_cluster=512
    model,_=build_our_model(Namespace(train_until_16x=False,train_refiner=False,
                           train_all_model=False,num_cluster=512),cfg,use_dinov2=True)
    weights=torch.load(args.weights,map_location='cpu',weights_only=True)
    weights,ignored=checkpoint_for_model(weights, model.state_dict().keys())
    (cache/"checkpoint_ignored_keys.json").write_text(json.dumps(ignored,indent=2)+"\n")
    model.load_state_dict(weights,strict=True)
    del weights
    model.eval().to('cuda')
    prior=[None,UniFlowMatchClassificationRefinement.from_pretrained(
        'infinity1096/UFM-Refine',revision=args.prior_revision).eval().to('cuda')]
    torch.set_num_threads(8)
    for state in args.states:
        items=manifest['groups'][state]
        for source in range(len(items)):
            dest=cache/state/f'{source}.npz'
            if dest.exists(): continue
            targets=[j for j in range(len(items)) if j!=source]
            started=time.time(); torch.cuda.reset_peak_memory_stats()
            result=run_model_test(model,{
                'query_img_path':str(args.input/items[source]['image']),
                'ref_img_paths':[str(args.input/items[j]['image']) for j in targets]},
                coarse_res_hw=(args.coarse,args.coarse),target_res_hw=(args.resolution,args.resolution),
                prematch_model=prior,prematch_model_name='ufm',num_cluster=512,
                upsample_preds=True,device='cuda')
            finest=result[min(result)]
            flow=finest['flow'][0].float().cpu().numpy()
            certainty=finest['certainty'][0,:,0].sigmoid().float().cpu().numpy()
            dest.parent.mkdir(parents=True,exist_ok=True)
            tmp=dest.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,flow=flow,certainty=certainty,targets=targets)
            tmp.replace(dest)
            record={'state':state,'source':items[source]['source'],
                    'seconds':time.time()-started,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                    'shape':list(flow.shape)}
            dest.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(record),flush=True)
            del result,finest,flow,certainty
            torch.cuda.empty_cache()


def reconstruct(args, manifest, state, cache, work):
    import pycolmap as pc
    items=manifest['groups'][state]
    images=args.input/Path(items[0]['image']).parent
    names=[Path(item['image']).name for item in items]
    predictions=[]
    for i in range(len(items)):
        with np.load(cache/state/f'{i}.npz') as d:
            predictions.append({j:(d['flow'][k],d['certainty'][k])
                                for k,j in enumerate(d['targets'].tolist())})
    indices=[KeypointIndex(args.cell_size) for _ in items]
    pairs={}; counts=[]
    for i,j in itertools.permutations(range(len(items)),2):
        flow,conf=predictions[i][j]; reverse,reverse_conf=predictions[j][i]
        src,dst,score=sample_pair(flow,conf,reverse,reverse_conf,items[i]['size'],items[j]['size'],
                                stride=args.stride,confidence=args.confidence,cycle_px=args.cycle_px)
        a,b=indices[i].add(src),indices[j].add(dst)
        key=(min(i,j),max(i,j))
        matches=np.column_stack([a,b] if i<j else [b,a])
        pairs.setdefault(key,[]).append((matches,score))
        counts.append({'source':names[i],'target':names[j],'accepted':len(a)})
        print(f'{state} {names[i]} -> {names[j]}: {len(a)} reciprocal matches',flush=True)
    del predictions
    database=work/'database.db'
    pc.Database.open(database).close()
    pc.import_images(database,images,camera_mode=pc.CameraMode.AUTO,
                     options=pc.ImageReaderOptions(camera_model='SIMPLE_RADIAL'))
    db=pc.Database.open(database)
    db_images={im.name:im for im in db.read_all_images()}
    for i,item in enumerate(items):
        im=db_images[names[i]]
        camera=db.read_camera(im.camera_id)
        if item['focal_35mm']:
            # Equivalent focal length scales with image diagonal (36x24 mm reference).
            params=camera.params.copy()
            params[0]=item['focal_35mm']*np.hypot(camera.width,camera.height)/np.hypot(36,24)
            camera.params=params; camera.has_prior_focal_length=True
            db.update_camera(camera)
        db.write_keypoints(im.image_id,indices[i].points())
    pair_names=[]
    for (i,j),chunks in pairs.items():
        matches=np.concatenate([c[0] for c in chunks]); scores=np.concatenate([c[1] for c in chunks])
        # Resolve conflicting pairwise assignments by descending confidence.
        used_a=set();used_b=set();keep=[]
        for k in np.argsort(-scores,kind='stable'):
            a,b=map(int,matches[k])
            if a in used_a or b in used_b: continue
            used_a.add(a);used_b.add(b);keep.append(k)
        matches=matches[keep].astype(np.uint32).reshape(-1,2)
        if len(matches)>=30:
            db.write_matches(db_images[names[i]].image_id,db_images[names[j]].image_id,matches)
            pair_names.append(f'{names[i]} {names[j]}')
    db.close()
    (work/'match_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    pairfile=work/'pairs.txt';pairfile.write_text('\n'.join(pair_names)+'\n')
    verification=pc.TwoViewGeometryOptions()
    verification.ransac.max_error=2.
    verification.ransac.random_seed=17
    pc.verify_matches(database,pairfile,options=verification)
    db=pc.Database.open(database)
    verified=[]
    for i,j in itertools.combinations(range(len(items)),2):
        id1,id2=db_images[names[i]].image_id,db_images[names[j]].image_id
        if db.exists_two_view_geometry(id1,id2):
            geometry=db.read_two_view_geometry(id1,id2)
            verified.append({'pair':[names[i],names[j]],'inliers':len(geometry.inlier_matches),
                             'config':str(geometry.config)})
    db.close()
    (work/'verified_pairs.json').write_text(json.dumps(verified,indent=2)+'\n')
    sparse=work/'sparse'; sparse.mkdir()
    options=pc.IncrementalPipelineOptions()
    options.min_model_size=3;options.num_threads=8;options.random_seed=17
    options.mapper.init_min_tri_angle=4.
    options.mapper.filter_max_reproj_error=2.
    options.mapper.abs_pose_max_error=4.
    options.triangulation.merge_max_reproj_error=2.
    options.triangulation.complete_max_reproj_error=2.
    models=pc.incremental_mapping(database,images,sparse,options=options)
    if not models: raise RuntimeError('COLMAP recovered no model; inspect verified_pairs.json and logs')
    model_id,model=max(models.items(),key=lambda kv:(kv[1].num_reg_images(),kv[1].num_points3D()))
    if model.num_reg_images()<3 or model.num_points3D()==0:
        raise RuntimeError('Fewer than three registered views or empty cloud')
    # COLMAP can merge nearby tracks that contain multiple observations in one
    # image. Such tracks overstate multi-view support; keep only unambiguous
    # points seen by at least three distinct cameras.
    raw=work/'raw_model';raw.mkdir();model.write(raw)
    rejected={'duplicate_image_observations':0,'fewer_than_three_views':0}
    for point_id,point in list(model.points3D.items()):
        image_ids=[e.image_id for e in point.track.elements]
        if len(set(image_ids)) != len(image_ids):
            rejected['duplicate_image_observations']+=1
            model.delete_point3D(point_id)
        elif len(image_ids)<3:
            rejected['fewer_than_three_views']+=1
            model.delete_point3D(point_id)
    if model.num_points3D()==0:
        raise RuntimeError('No unambiguous three-view points survived')
    pc.bundle_adjustment(model)
    model.update_point_3d_errors()
    model.write(sparse/str(model_id))
    model.export_PLY(work/'sparse.ply')
    errors=np.array([p.error for p in model.points3D.values()])
    lengths=np.array([p.track.length() for p in model.points3D.values()])
    registered=sorted(model.images[k].name for k in model.reg_image_ids())
    result={'material':'box','state':state,'status':'success' if len(registered)==len(items) else 'partial',
            'input_images':len(items),'registered_images':len(registered),'registered_names':registered,
            'unregistered_names':sorted(set(names)-set(registered)),
            'sparse_points':model.num_points3D(),'rejected_tracks':rejected,'selected_model':str(sparse/str(model_id)),
            'sparse_cloud':str(work/'sparse.ply'),'mean_reprojection_error_px':model.compute_mean_reprojection_error(),
            'median_point_error_px':float(np.median(errors)),'p95_point_error_px':float(np.quantile(errors,.95)),
            'mean_track_length':float(lengths.mean()),'track_length_histogram':{
                str(int(n)):int(np.sum(lengths==n)) for n in np.unique(lengths)},
            'cameras':{str(k):{'model':c.model_name,'width':c.width,'height':c.height,'params':c.params.tolist()}
                       for k,c in model.cameras.items()},
            'units':'arbitrary SfM scale; states not aligned',
            'adapter':'custom reciprocal filtering + spatial observation consolidation + COLMAP'}
    (work/'sparse_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--mvroma',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--states',nargs='+',choices=['before','after'],default=['before','after'])
    p.add_argument('--coarse',type=int,default=560)
    p.add_argument('--resolution',type=int,default=840)
    p.add_argument('--stride',type=int,default=4)
    p.add_argument('--confidence',type=float,default=.8)
    p.add_argument('--cycle-px',type=float,default=2.)
    p.add_argument('--cell-size',type=float,default=1.)
    p.add_argument('--geometry-only',action='store_true')
    p.add_argument('--prior-revision',default='89cc70107b483170ebfc908630e6f40ca1b77315')
    args=p.parse_args()
    if args.output.exists(): p.error('Use a fresh output directory; match cache can be reused')
    if args.coarse<=0 or args.coarse%14 or args.resolution<=0 or args.resolution%14:
        p.error('Model dimensions must be positive multiples of 14')
    if args.stride<1 or args.cell_size<=0 or args.cycle_px<=0 or not 0<=args.confidence<=1:
        p.error('Invalid sampling/filter parameters')
    manifest=json.loads((args.input/'manifest.json').read_text())
    for state in args.states:
        for item in manifest['groups'][state]:
            if sha256(args.input/item['image'])!=item['image_sha256']:
                raise RuntimeError(f'Image checksum mismatch: {item["image"]}')
    ufm=args.mvroma.parent/'UFM'
    signature={'manifest_sha256':sha256(args.input/'manifest.json'),
               'weights_sha256':sha256(args.weights),'mvroma':source_identity(args.mvroma),
               'ufm':source_identity(ufm),'uniception':source_identity(ufm/'UniCeption'),
               'prior_revision':args.prior_revision,
               'inference_sha256':hashlib.sha256((inspect.getsource(infer)+inspect.getsource(checkpoint_for_model)).encode()).hexdigest(),
               'torch':package_version('torch'),'torchvision':package_version('torchvision'),
               'coarse':args.coarse,'resolution':args.resolution,'num_cluster':512}
    validate_cache(args.cache,signature)
    args.output.mkdir(parents=True)
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.output/'config.json').write_text(json.dumps(dict(settings=settings,inference=signature),indent=2)+'\n')
    if not args.geometry_only: infer(args,manifest,args.cache)
    results=[]
    for state in args.states:
        work=args.output/'box'/state; work.mkdir(parents=True)
        try: result=reconstruct(args,manifest,state,args.cache,work)
        except Exception as exc:
            traceback.print_exc();result={'material':'box','state':state,'status':'failed','error':str(exc)}
        results.append(result)
        (work/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        (args.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    return int(any(r['status']!='success' for r in results))

if __name__=='__main__':sys.exit(main())
