import torch
import gc
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
# Example of saving the latents as meshes.
from shap_e.util.notebooks import decode_latent_mesh
from fastapi import FastAPI, Request, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import boto3
import aspose.threed as a3d
import time
from numba import cuda 



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class Item(BaseModel):
    prompt: str
    models_nbr:int


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = boto3.Session(aws_access_key_id='XXXXXXXX-ask-Devops',
                        aws_secret_access_key='XXXXXXX-ask-Devops')


s3 = session.resource('s3')


@app.post('/')
async def generate_3d(item: Item, request: Request):

    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    file_name = request.headers.get('filename').replace(' ', '_')
    list_models=[]
    batch_size = item.models_nbr
    guidance_scale = 15.0
    prompt = item.prompt
    time_in_sec=str(time.time())

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    render_mode = 'nerf' # you can change this to 'stf'
    size = 128 # this is the size of the renders; higher values take longer to render.
    cameras = create_pan_cameras(size, device)
    for i, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latent).tri_mesh()
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)
        with open(f'result/{str(file_name)}_{time_in_sec}_{i}.ply', 'wb') as f:
            t.write_ply(f)

            scene = a3d.Scene.from_file(f'result/{str(file_name)}_{time_in_sec}_{i}.ply')
            scene.save(f'result/{str(file_name)}_{time_in_sec}_{i}.glb')

            images[0].save(f'result/{str(file_name)}_{time_in_sec}_{i}.gif', save_all=True, append_images=images, duration=200, loop=0)

            s3 = session.resource('s3')

            s3.meta.client.upload_file(Filename=f'result/{str(file_name)}_{time_in_sec}_{i}.glb', Bucket='abxr-backend', Key=f'media/pointE/{str(file_name)}_{time_in_sec}_{i}.glb')
            s3.meta.client.upload_file(Filename=f'result/{str(file_name)}_{time_in_sec}_{i}.gif', Bucket='abxr-backend', Key=f'media/pointE/{str(file_name)}_{time_in_sec}_{i}.gif')

            list_models.append({'model':f'https://abxr-backend.s3.amazonaws.com/media/pointE/{str(file_name)}_{time_in_sec}_{i}.glb',
                                'gif':f'https://abxr-backend.s3.amazonaws.com/media/pointE/{str(file_name)}_{time_in_sec}_{i}.gif'})

    del xm
    del model
    del diffusion
    del latents
    torch.cuda.empty_cache()
    gc.collect()
    device = cuda.get_current_device()
    device.reset()
    return list_models
