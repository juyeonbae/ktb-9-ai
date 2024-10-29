from diffusers import StableDiffusionInpaintPipeline

def load_stable_diffusion_inpaint():
    pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
    return pipeline

sd_inpaint_pipeline = load_stable_diffusion_inpaint()
