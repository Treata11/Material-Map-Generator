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

    # print("Output Images RGB values")
    # for coords in pixel_coords:
    #     x, y = coords
    #     rgb_values = output[y, x]  # Note: output[y, x] gives you [R, G, B]
    #     # Print RGB values with high precision
    #     print(f"Pixel{coords}: R={rgb_values[0]:.6f}, G={rgb_values[1]:.6f}, B={rgb_values[2]:.6f}")

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
    # break

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
normal_imgShape = (1, 3, imgSize, imgSize) 
roughness_imgShape = (1, 1, imgSize, imgSize) 
# imgShape = (imgSize, imgSize, 3) 
example_input_normal = torch.rand(*normal_imgShape)  # Example input needed for tracing
example_input_roughness = torch.rand(*roughness_imgShape)

# α = 0.0889850649
# β = 0.1348233757
# γ = 0.0566280158
α = 0.0
β = 0.0
γ = 0.0
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
    traced_model = torch.jit.trace(torch_model, example_input_normal, strict=True) 
    # traced_model.save(NORMAL_MAP_MODEL) # Optional, can pass traced model directly to converter.
      
    image_input=[ct.ImageType(
        name="imageTexture", shape=example_input_normal.shape, 
        color_layout=ct.colorlayout.BGR,
        bias=[-0.485/0.229,-0.456/0.224,-0.406/0.225], scale=1.0/255.0/0.226
    )]
    tensor_input= [ct.TensorType(name="imageTexture", shape=example_input_normal.shape, dtype=np.float16)]

    image_output=[ct.ImageType(name="normalMap", color_layout=ct.colorlayout.BGR, channel_first=None)]
    output_tensor = [ct.TensorType(name="normalMap", dtype=np.float32)]

    coreml_model = ct.convert( 
        traced_model, 
        inputs=image_input, 
        # 'channel_first' must be None for an output of ImageType
        outputs=output_tensor, 
        source='pytorch',  
        convert_to='mlprogram', 
        compute_precision=ct.precision.FLOAT16,
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
    #     example_input_normal = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)  # Convert to correct shape
    traced_model = torch.jit.trace(torch_model, example_input_normal, strict=True) 
    # traced_model.save(OTHER_MAP_MODEL) # Optional, can pass traced model directly to converter.

    image_input=ct.ImageType(
        name="imageTexture", shape=example_input_normal.shape, 
        color_layout=ct.colorlayout.BGR, 
        bias=[-0.485/0.229,-0.456/0.224,-0.406/0.225], scale=1.0/255.0/0.226
    )
    coreml_model = ct.convert( 
        traced_model, 
        inputs=[image_input], 
        outputs=[
            ct.ImageType(name="roughnessMap", color_layout=ct.colorlayout.GRAYSCALE), # GRAYSCALE_FLOAT16
        ], 
        source='pytorch',  
        convert_to='mlprogram', 
        debug=True,
        minimum_deployment_target=ct.target.macOS14
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
        # continue
        convert_normal_map_generator(model, names[idx])
    elif idx == 1:
        # break
        convert_roughness_displacement_generator(model, names[idx])


# def post_process_outputs(rlts, ishiiruka_texture_encoder=False):
#     normal_map = rlts[0]
#     roughness = rlts[1][:, :, 1]
#     displacement = rlts[1][:, :, 0]

#     if ishiiruka_texture_encoder:
#         r = 255 - roughness
#         g = normal_map[:, :, 1]
#         b = displacement
#         a = normal_map[:, :, 2]
#         output = cv2.merge((b, g, r, a))
#         return output
#     else:
#         return {
#             "normal_map": normal_map,
#             "roughness": roughness,
#             "displacement": displacement
#         }


# MARK: --- --- --- --- --- Predict --- --- --- --- ---

# def preprocess_for_coreml(img):
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
#     # img = cv2.resize(img, (256, 256))  
#     img = img / 255.0  # Normalize
#     return img

# Load CoreML model
model_name = 'NormalMapGenerator-CX-Lite_200000_G'
coreml_model = ct.models.MLModel(f"{model_name}.mlpackage")

def convert_multiarray_output_to_image(spec, feature_name, feature_shape, is_bgr=False):
    for output in spec.description.output:
        if output.name != feature_name:
            print('false feature_name')
            continue

        array_shape = feature_shape
        _, _, height, width = array_shape
        from coremltools.proto import FeatureTypes_pb2 as ft
        output.type.imageType.colorSpace = ft.ImageFeatureType.ColorSpace.Value('BGR')
        output.type.imageType.width = width
        output.type.imageType.height = height
        # output.type.imageType.

        # channels, height, width = tuple(output.type.multiArrayType.shape)
        # # Set image shape
        # output.type.imageType.width = width 
        # output.type.imageType.height = height

spec = coreml_model.get_spec()
print(spec.description.output[0].type)

convert_multiarray_output_to_image(spec, "normalMap", (1, 4, imgSize, imgSize))

print(spec.description.output[0].type)
spec.description.output[0].type

ct.utils.save_spec(spec, 'TensorsToImage.mlmodel')



from PIL import Image  # Import PIL for image 

for idx, path in enumerate(images, 1):
    # break

    base = os.path.splitext(os.path.relpath(path, input_folder))[0]
    output_dir = os.path.dirname(os.path.join(output_folder, base))
    os.makedirs(output_dir, exist_ok=True)
    print(idx, base)

    # Read image
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not loaded. Check the path.")

    # # Preprocess image for CoreML model
    # coreml_input = preprocess_for_coreml(img)

    # Get output from CoreML model  (Done in scale and bias)
    pil_image = Image.fromarray(img)  # Convert NumPy array to PIL Image
    coreml_output = coreml_model.predict({'imageTexture': pil_image})

    coreml_output_image = coreml_output['normalMap']

    # np.set_printoptions(threshold=np.inf) 
    print(f"CoreML Output: {coreml_output_image}")
    # print(f"CoreML Output Image: {coreml_output_image.size}, Mode: {coreml_output_image.mode}")

    ## Post Processing
    coreml_output_array = np.array(coreml_output_image)  # Convert PIL Image to NumPy array
    # coreml_output_array = coreml_output_array[[2, 1, 0], :, :] # RGB to BGR
    # coreml_output_array = np.transpose(coreml_output_array, (1, 2, 0)) 
    # print(f"Output Array Shape: {coreml_output_array.shape}, Dtype: {coreml_output_array.dtype}")



    # MARK: - Compare numerical output
    print("\n\n- Compare Pytorch and CoreML outputs:")

    input = img * 1. / np.iinfo(img.dtype).max
    input = input[:, :, [2, 1, 0]]
    input = torch.from_numpy(np.transpose(input, (2, 0, 1))).float()
    img_LR = input.unsqueeze(0)
    img_LR = img_LR.to(device)


    # coreml_output_array = np.clip(coreml_output_array, 0, 1)  # Ensure values are between 0 and 1
    # output_array = (output_array * 255).astype(np.uint8)  # Scale to [0, 255] and convert to uint8

    output_file_path = os.path.join(output_folder, f'{base}_coreML_normal_map.png')
    cv2.imwrite(output_file_path, cv2.cvtColor(coreml_output_array, cv2.COLOR_RGBA2BGR))  # If output is RGBA

    print(f"Saved output image: {output_file_path}")

    pytorchModel = models[0] # Normal-map model
    # torch_output = pytorchModel(img_LR).data.squeeze(0).float().cpu().clamp_(0, 1).numpy()
    torch_output = pytorchModel(img_LR).data.cpu().numpy()
    np.testing.assert_allclose(torch_output, coreml_output_array)
    print("-------------------------------------------------\n")
