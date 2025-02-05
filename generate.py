import argparse
import os

import cv2
import numpy as np
import torch
import sys

# upscale & crop
import utils.imgops as ops
import utils.architecture.architecture as arch

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='input', help='Input folder')
parser.add_argument('--output', default='output', help='Output folder')
parser.add_argument('--reverse', help='Reverse Order', action="store_true")
parser.add_argument('--tile_size', default=512,
                    help='Tile size for splitting', type=int)
parser.add_argument('--seamless', action='store_true',
                    help='Seamless upscaling')
parser.add_argument('--mirror', action='store_true',
                    help='Mirrored seamless upscaling')
parser.add_argument('--replicate', action='store_true',
                    help='Replicate edge pixels for padding')
parser.add_argument('--cpu', action='store_true',
                    help='Use CPU instead of CUDA')
parser.add_argument('--ishiiruka', action='store_true',
                    help='Save textures in the format used in Ishiiruka Dolphin material map texture packs')
parser.add_argument('--ishiiruka_texture_encoder', action='store_true',
                    help='Save textures in the format used by Ishiiruka Dolphin\'s Texture Encoder tool')
args = parser.parse_args()

if not os.path.exists(args.input):
    print('Error: Folder [{:s}] does not exist.'.format(args.input))
    sys.exit(1)
elif os.path.isfile(args.input):
    print('Error: Folder [{:s}] is a file.'.format(args.input))
    sys.exit(1)
elif os.path.isfile(args.output):
    print('Error: Folder [{:s}] is a file.'.format(args.output))
    sys.exit(1)
elif not os.path.exists(args.output):
    os.mkdir(args.output)

device = torch.device('cpu' if args.cpu else 'cuda')

input_folder = os.path.normpath(args.input)
output_folder = os.path.normpath(args.output)

NORMAL_MAP_MODEL = 'utils/models/1x_NormalMapGenerator-CX-Lite_200000_G.pth'
OTHER_MAP_MODEL = 'utils/models/1x_FrankenMapGenerator-CX-Lite_215000_G.pth'

# Initialize a list to store the images
images_to_process = []

## -
def process(img, model):
    # print(f"def process before; img Shape: {img.shape}") 
    # print("def process before; img Data Type:", img.dtype)
    # print("def process before; Sample img Pixel Values:", img[0, 0]) # Top left RGBs 
    # print(f"def process before; img dtype: {np.info(img.dtype)}") 
    # print(f"def process before; img : {img[:, :, [2, 1, 0]]}") 

    # Preprocess
    img = img * 1. / np.iinfo(img.dtype).max
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # print(f"def process after; img Shape: {img.shape}") 
    # print("def process after; img Data Type:", img.dtype)
    # print("def process after; Sample img Pixel Values:", img[0, 0]) # Top left RGBs 

    ## Store the image in the list before processing
    images_to_process.append(img)
    
    # Write the tensor image on disk
    img_name = '{:s}_Def_Proecess_Img.png'.format(base)
    # cv2.imwrite(os.path.join(output_folder, img_name), img.numpy())

    # Post Processes
    output = model(img_LR).data.squeeze(
        0).float().cpu().clamp_(0, 1).numpy()
    output = output[[2, 1, 0], :, :]
    output = np.transpose(output, (1, 2, 0))
    output = (output * 255.).round()
    return output

def load_model(model_path):
    global device
    state_dict = torch.load(model_path, weights_only=False)
    model = arch.RRDB_Net(3, 3, 32, 12, gc=32, upscale=1, norm_type=None, act_type='leakyrelu',
                            mode='CNA', res_scale=1, upsample_mode='upconv')
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    return model.to(device)

images=[]
for root, _, files in os.walk(input_folder):
    for file in sorted(files, reverse=args.reverse):
        if file.split('.')[-1].lower() in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'tga', 'heic', 'heif', 'dng']:
            images.append(os.path.join(root, file))

models = [
    # NORMAL MAP
    load_model(NORMAL_MAP_MODEL), 
    # ROUGHNESS/DISPLACEMENT MAPS
    load_model(OTHER_MAP_MODEL)
    ]

for idx, path in enumerate(images, 1):
    base = os.path.splitext(os.path.relpath(path, input_folder))[0]
    output_dir = os.path.dirname(os.path.join(output_folder, base))
    os.makedirs(output_dir, exist_ok=True)
    print(idx, base)
    # read image
    try: 
        img = cv2.imread(path, cv2.cv2.IMREAD_COLOR)
    except:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
    # Seamless modes
    if args.seamless:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_WRAP)
    elif args.mirror:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REFLECT_101)
    elif args.replicate:
        img = cv2.copyMakeBorder(img, 16, 16, 16, 16, cv2.BORDER_REPLICATE)

    img_height, img_width = img.shape[:2]

    # Whether or not to perform the split/merge action
    do_split = img_height > args.tile_size or img_width > args.tile_size

    # Process the Image ##
    if do_split:
        rlts = ops.esrgan_launcher_split_merge(img, process, models, scale_factor=1, tile_size=args.tile_size)
    else:
        rlts = [process(img, model) for model in models]

    if args.seamless or args.mirror or args.replicate:
        rlts = [ops.crop_seamless(rlt) for rlt in rlts]

    normal_map = rlts[0]
    roughness = rlts[1][:, :, 1]
    displacement = rlts[1][:, :, 0]

    if args.ishiiruka_texture_encoder:
        r = 255 - roughness
        g = normal_map[:, :, 1]
        b = displacement
        a = normal_map[:, :, 2]
        output = cv2.merge((b, g, r, a))
        cv2.imwrite(os.path.join(output_folder, '{:s}.mat.png'.format(base)), output)
    else:
        normal_name = '{:s}.nrm.png'.format(base) if args.ishiiruka else '{:s}_Normal.png'.format(base)
        cv2.imwrite(os.path.join(output_folder, normal_name), normal_map)

        rough_name = '{:s}.spec.png'.format(base) if args.ishiiruka else '{:s}_Roughness.png'.format(base)
        rough_img = 255 - roughness if args.ishiiruka else roughness
        cv2.imwrite(os.path.join(output_folder, rough_name), rough_img)

        displ_name = '{:s}.bump.png'.format(base) if args.ishiiruka else '{:s}_Displacement.png'.format(base)
        cv2.imwrite(os.path.join(output_folder, displ_name), displacement)


import coremltools as ct

## TODO: Check if `Normalization and Scaling` operations took place before conversion to coreML

# TODO: Make flexible
imgSize = 256 
imgShape = (1, 3, imgSize, imgSize) 
# imgShape = (imgSize, imgSize, 3) 
example_input = torch.rand(*imgShape)  # Example input needed for tracing

# https://apple.github.io/coremltools/docs-guides/source/image-inputs.html#preprocessing-for-torch
scale = 1/(0.226*255.0)
bias = [- 0.485/(0.229) , - 0.456/(0.224), - 0.406/(0.225)]

# scale = 200/(0.226*255.0)
# bias = [1, 1, 1]  # No bias for each channel
# bias = [0.485/(0.229) , 0.456/(0.224), 0.406/(0.225)]

for img in images_to_process:
    print(f"img Shape: {img.shape}") 
    print("img Data Type:", img.dtype)
    print("Sample img Pixel Values:", img[0, 0]) # Top left RGBs 

print(f"example_input Shape: {example_input.shape}") 
print("example_input Data Type:", example_input.dtype)
print("Sample example_input Pixel Values:", example_input[0, 0]) # Top left RGBs

def convert_normal_map_generator(torch_model, model_name): 
    traced_model = torch.jit.trace(torch_model, example_input) 
    # traced_model.save(NORMAL_MAP_MODEL) # Optional, can pass traced model directly to converter.
      
    image_input=ct.ImageType(
        name="Image_Texture", shape=example_input.shape, 
        color_layout=ct.colorlayout.BGR, scale=scale, bias=bias
    )
    coreml_model = ct.convert( 
        traced_model, 
        inputs=[image_input], 
        outputs=[ct.ImageType(name="Normal_Map", color_layout=ct.colorlayout.RGB)], 
        source='pytorch',  
        convert_to='mlprogram', 
        minimum_deployment_target=ct.target.macOS13 
    )  
    coreml_model.save(f"{model_name}.mlpackage") 
    print(f"Model saved as {model_name}.mlpackage") 
    loaded_model = ct.models.MLModel(f"{model_name}.mlpackage") 
    loaded_model.short_description = "Creates Normal map from a given texture for Physically-Based Rendering"

def convert_roughness_displacement_generator(torch_model, model_name): 
    # for img in images_to_process:
    #     imgShape = img.shape[:2]
    #     example_input = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # Convert to correct shape
    traced_model = torch.jit.trace(torch_model, example_input) 
    # traced_model.save(OTHER_MAP_MODEL) # Optional, can pass traced model directly to converter.

    image_input=ct.ImageType(
        name="Image_Texture", shape=example_input.shape, 
        color_layout=ct.colorlayout.BGR, scale=scale, bias=bias
    )
    coreml_model = ct.convert( 
        traced_model, 
        inputs=[image_input], 
        outputs=[
            ct.ImageType(name="Other_Map", color_layout=ct.colorlayout.RGB), # GRAYSCALE_FLOAT16
        ], 
        source='pytorch',  
        convert_to='mlprogram', 
        minimum_deployment_target=ct.target.macOS13 
    )  
    coreml_model.save(f"{model_name}.mlpackage") 
    print(f"Model saved as {model_name}.mlpackage") 
    loaded_model = ct.models.MLModel(f"{model_name}.mlpackage") 
    loaded_model.short_description = "Creates Roughness and Displacement maps from a given texture for Physically-Based Rendering"


# Convert each model to MLPackage/CoreML
for idx, model in enumerate(models): 
    names = ['1x_NormalMapGenerator-CX-Lite_200000_G', '1x_FrankenMapGenerator-CX-Lite_215000_G'] 
    if idx == 0:
        convert_normal_map_generator(model, names[idx])
    elif idx == 1:
        convert_roughness_displacement_generator(model, names[idx])


# Lack of Post-Process in CoreML Tools
def post_process_outputs(rlts, ishiiruka_texture_encoder=False):
    normal_map = rlts[0]
    roughness = rlts[1][:, :, 1]
    displacement = rlts[1][:, :, 0]

    if ishiiruka_texture_encoder:
        r = 255 - roughness
        g = normal_map[:, :, 1]
        b = displacement
        a = normal_map[:, :, 2]
        output = cv2.merge((b, g, r, a))
        return output
    else:
        return {
            "normal_map": normal_map,
            "roughness": roughness,
            "displacement": displacement
        }

## Make a prediction using CoreML ##
# for idx, path in enumerate(images, 1):
#     base = os.path.splitext(os.path.relpath(path, input_folder))[0]
#     output_dir = os.path.dirname(os.path.join(output_folder, base))
#     os.makedirs(output_dir, exist_ok=True)
#     print(idx, base)
#     # read image
#     try: 
#         test_image = cv2.imread(path, cv2.cv2.IMREAD_COLOR)
#     except:
#         test_image = cv2.imread(path, cv2.IMREAD_COLOR)
# out_dict = model.predict({input_name: test_image})
# output = model.predict({"colorImage" : test_image})["colorOutput"]
# display(output)
# output.save("Prediction-Result.png")