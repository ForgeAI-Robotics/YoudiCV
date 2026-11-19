"""Read-only environment check; does not install or change any packages."""
import importlib
import json
import platform

report={'python':platform.python_version(),'architecture':platform.machine(),'packages':{}}
for name in ['torch','torchvision','ultralytics','cv2','numpy']:
    try:
        module=importlib.import_module(name)
        report['packages'][name]={'version':getattr(module,'__version__','unknown'),'available':True}
        if name=='torch':
            report['cuda_available']=module.cuda.is_available()
            report['torch_cuda_version']=module.version.cuda
    except Exception as exc:
        report['packages'][name]={'available':False,'error':str(exc)}
print(json.dumps(report,ensure_ascii=False,indent=2))
