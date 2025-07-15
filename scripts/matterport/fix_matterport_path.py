import os
import tqdm
from pxr import Usd, UsdShade, Sdf, Ar

OLD_PREFIX = "./textures"        

main_directory = "matterport_usd"

for subfolder in tqdm.tqdm(os.listdir(main_directory)):
    subfolder_path = os.path.join(main_directory, subfolder)
    if os.path.isdir(subfolder_path):
        subfolder_path = os.path.join(main_directory, subfolder, subfolder + ".usd")
        NEW_PREFIX = os.path.join(os.path.abspath(main_directory), subfolder, "textures")
        
        stage = Usd.Stage.Open(subfolder_path)
        root_layer = stage.GetRootLayer()  
        
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Shader":
                continue

            shader = UsdShade.Shader(prim)

            if hasattr(shader, "GetInputs"):
                inputs = shader.GetInputs()
            else:                             
                inputs = [shader.GetInput(n) for n in shader.GetInputNames()]

            for inp in inputs:
                attr = inp.GetAttr()
                if attr.GetTypeName() != Sdf.ValueTypeNames.Asset:
                    continue

                asset_path = inp.Get()
                if asset_path is None:
                    continue

                old_path = asset_path.path
                
                if old_path.startswith(OLD_PREFIX):
                    new_path = old_path.replace(OLD_PREFIX, NEW_PREFIX, 1)
                    inp.Set(Sdf.AssetPath(new_path))
        
        root_layer.Export(os.path.join(main_directory, subfolder, "fixpath.usd")) 