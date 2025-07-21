import defaults.bootstrap

if __name__ == "__main__":
    pass
    # from executors.example_text_to_image import (
    #     T2IExecutor,
    #     T2IHireFixExecutor,
    #     T2ILoRAExecutor,
    # )

    # executor = T2IExecutor()
    # executor = T2ILoRAExecutor()
    # executor = T2IHireFixExecutor()

    # executor(
    #     pos_text="1girl, solo, long hair, ponytail, blue hair, blue eyes, open mouth, nose blush, shy, white background, masterpiece, best quality, very aesthetic, absurdres, anime coloring,",
    #     neg_text="wings, nsfw, low quality, worst quality, normal quality",
    #     width=1024,
    #     height=1024,
    #     steps=30,
    #     cfg=8.0,
    #     seed=124444,
    # )

    # from executors.example_image_text_to_image import IT2IExecutor

    # executor = IT2IExecutor()
    # executor(
    #     "./examples/sample.png",
    #     "a photo of flat color character,masterpiece, expressionless, Extremely detailed high quality photo, 8k resolution, anime style, key visual, studio anime, white background",
    #     "photo, deformed, black and white, realism, disfigured, low contrast, Deviantart, jpeg , (worst quality, low quality, normal quality, lowres, low details, oversaturated, undersaturated, overexposed, underexposed, grayscale, bw, bad photo, bad photography, bad art:1.4), (watermark, signature, text font, username, error, logo, words, letters, digits, autograph, trademark, name:1.2), (blur, blurry, grainy), morbid, ugly, asymmetrical, mutated malformed, mutilated, poorly lit, bad shadow, draft, cropped, out of frame, cut off, censored, jpeg artifacts, out of focus, glitch, duplicate, (airbrushed, cartoon, anime, semi-realistic, cgi, render, blender, digital art, manga, amateur:1.3), (3D ,3D Game, 3D Game Scene, 3D Character:1.1), (bad hands, bad anatomy, bad body, bad face, bad teeth, bad arms, bad legs, deformities:1.3),bad hands, bad fingers,R18",
    #     1024,
    #     1024,
    #     40,
    #     6.0,
    #     12419249,
    # )
    # from executors.example_ipadapter_image_text_to_image import IPAdapterIT2I
    # executor = IPAdapterIT2I()
    # executor(
    #     image_path="./examples/reference.png",
    #     ppromt="1girl, solo, long hair, ponytail, blue hair, blue eyes, open mouth, nose blush, shy, white background, masterpiece, best quality, very aesthetic, absurdres, anime coloring,",
    #     nprompt="wings, nsfw, low quality, worst quality, normal quality",
    #     width=1024,
    #     height=1024,
    #     steps=30,
    #     cfg=8.0,
    #     seed=124444,
    # )

    from executors.example_flux import FluxSimpleExectuor, FluxUnderVRAM12GB, FluxInpainting

    # executor = FluxSimpleExectuor()
    # executor()
    
    executor = FluxInpainting()
    executor(
        image_path="./examples/flux_fill_inpaint_example.png",
        mask_path="./examples/flux_fill_inpaint_example_mask.png",
        # nprompt="",
        # steps=20,
        # cfg=8.0,
        # seed=124444,
    )
