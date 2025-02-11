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
    pixel_coords = [(0, 0), (1, 1), (63, 63), (64, 64), (127, 127), (128, 128), (255, 255)]

    # Preprocess
    img = img * 1. / np.iinfo(img.dtype).max
    img = img[:, :, [2, 1, 0]]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    # Post Processes
    output = model(img_LR).data.squeeze(
        0).float().cpu().clamp_(0, 1).numpy()
    output = output[[2, 1, 0], :, :] # RGB to BGR
    output = np.transpose(output, (1, 2, 0)) # Reshape from (C, H, W) to (H, W, C)

    print("Output Images RGB values")
    for coords in pixel_coords:
        x, y = coords
        rgb_values = output[y, x]  # Note: output[y, x] gives you [R, G, B]
        # Print RGB values with high precision
        print(f"Pixel{coords}: R={rgb_values[0]:.6f}, G={rgb_values[1]:.6f}, B={rgb_values[2]:.6f}")

    output = (output * 255.).round()

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
# from coremltools.proto import NeuralNetwork_pb2

## TODO: Check if `Normalization and Scaling` operations took place before conversion to coreML

# TODO: Make flexible
imgSize = 256 
imgShape = (1, 3, imgSize, imgSize) 
# imgShape = (imgSize, imgSize, 3) 
example_input = torch.rand(*imgShape)  # Example input needed for tracing

α = 0.0889850649
β = 0.1348233757
γ = 0.0566280158
# https://apple.github.io/coremltools/docs-guides/source/image-inputs.html#preprocessing-for-torch
scale = 1/(0.226*255.0)
bias = [- (0.485 + α)/(0.229) , - (0.456 + β)/(0.224), - (0.406 + γ)/(0.225)]


# --------------------------------------------------
# from coremltools.converters.mil import Builder as mb
# from coremltools.converters.mil import register_torch_op

# # Post Processing with Custom Layers
# @register_torch_op
# def upsample_bicubic2d(context, node):
#     a = context[node.inputs[0]]
#     align_corners = context[node.inputs[2]].val
#     scale = context[node.inputs[3]]
#     scale_h = scale.val[0]
#     scale_w = scale.val[1]

#     x = mb.upsample_bilinear(x=a, scale_factor_height=scale_h, scale_factor_width=scale_w, align_corners=align_corners, name=node.name)
#     context.add(x)


# from coremltools.proto import NeuralNetwork_pb2

# def convert_lambda(layer):
#     # Only convert this Lambda layer if it is for our swish function.
#     if layer.function == swish:
#         params = NeuralNetwork_pb2.CustomLayerParams()

#         # The name of the Swift or Obj-C class that implements this layer.
#         params.className = "Swish"

#         # The desciption is shown in Xcode's mlmodel viewer.
#         params.description = "A fancy new activation function"

#         return params
#     else:
#         return None


# ---------------------------------



def convert_normal_map_generator(torch_model, model_name): 
    traced_model = torch.jit.trace(torch_model, example_input, strict=True) 
    # traced_model.save(NORMAL_MAP_MODEL) # Optional, can pass traced model directly to converter.
      
    image_input=ct.ImageType(
        name="imageTexture", shape=example_input.shape, 
        color_layout=ct.colorlayout.BGR, # no effects upon changing to BGR
        bias=[-0.485/0.229,-0.456/0.224,-0.406/0.225], scale=1.0/255.0/0.226
    )
    coreml_model = ct.convert( 
        traced_model, 
        inputs=[image_input], 
        outputs=[ct.ImageType(name="normalMap", color_layout=ct.colorlayout.BGR)], 
        # add_custom_layers=True, # deprecated
        # custom_conversion_functions={ "Lambda": convert_lambda }, # deprecated
        source='pytorch',  
        convert_to='mlprogram', 
        compute_precision=ct.precision.FLOAT32,
        debug=True,
        minimum_deployment_target=ct.target.macOS14 
    )  
    coreml_model.save(f"{model_name}.mlpackage") 
    print(f"Model saved as {model_name}.mlpackage") 
    loaded_model = ct.models.MLModel(f"{model_name}.mlpackage") 
    loaded_model.short_description = "Creates Normal map from a given texture for Physically-Based Rendering"


# - TODO: Complete
def convert_roughness_displacement_generator(torch_model, model_name): 
    # for img in images_to_process:
    #     imgShape = img.shape[:2]
    #     example_input = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # Convert to correct shape
    traced_model = torch.jit.trace(torch_model, example_input, strict=False) 
    # traced_model.save(OTHER_MAP_MODEL) # Optional, can pass traced model directly to converter.

    image_input=ct.ImageType(
        name="imageTexture", shape=example_input.shape, 
        color_layout=ct.colorlayout.BGR, scale=scale, bias=bias
    )
    coreml_model = ct.convert( 
        traced_model, 
        inputs=[image_input], 
        outputs=[
            ct.ImageType(name="frankenMap", color_layout=ct.colorlayout.RGB), # GRAYSCALE_FLOAT16
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
    names = ['NormalMapGenerator-CX-Lite_200000_G', 'FrankenMapGenerator-CX-Lite_215000_G'] 
    if idx == 0:
        # break
        convert_normal_map_generator(model, names[idx])
    elif idx == 1:
        break
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