import os
import math
import numpy as np
from PIL import Image
import torch

from map.lseg.additional_utils.models import resize_image, pad_image, crop_image

def get_lseg_feats(
    model,
    image,
    labels,
    crop_size,
    base_size,
    transform,
    norm_mean,
    norm_std,
):
    # Step0. load a image
    # image = Image.open(img_path)    
    # image = np.array(image)                                                     # 360, 480, 3       - H, W, D

    # Step0-1. preprocess
    # vis_image = image.copy()
    image = transform(image).unsqueeze(0).cuda()                                # 1, 3, 360, 480    - B, D, H, W
    img = image[0].permute(1, 2, 0)                                             # 360, 480, 3       - H, W, D
    img = img * 0.5 + 0.5

    stride_rate = 2.0 / 3.0
    stride = int(crop_size * stride_rate)

    batch, _, h, w = image.size()
    with torch.cuda.device_of(image):
        lseg_scores = image.new().resize_(batch,len(labels),h,w).zero_().cuda() # 1, 5, 360, 480

    # long_size = int(math.ceil(base_size * scale))
    long_size = base_size                           # scale 미존재
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height

    '''
        width = long_size = base_size = 520
        height = 390
        short_size = height = 390

        args.crop_size = 480                                # if dataset != 'citys'
        args.base_size = 520  
    '''
    # resize image to current size (1, 3, 390, 520)                         
    cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})
    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean,
                            norm_std, crop_size)
        with torch.no_grad():
            # outputs = module_inference(self.module, pad_img, label_set, self.flip)
            # 해당 결과가 모델 학습에 반영치 않토록 함으로써 output 출력
            vlmaps_outputs, lseg_outputs = model(pad_img, labels)                             # pixel_encoding, out
        vlmaps_outputs = crop_image(vlmaps_outputs, 0, height, 0, width)
        lseg_outputs = crop_image(lseg_outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed - short_size 를 crop_size(480) 에 맞춰서 padding 추가
            pad_img = pad_image(cur_img, norm_mean,                                            # 1, 3, 480, 520
                                norm_std, crop_size)
        else:
            pad_img = cur_img
        _,_,ph,pw = pad_img.shape #.size()                                                     # 1, 3, 480, 520
        assert(ph >= height and pw >= width)
        # grid forward and normalize
        h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
        '''
            image: (1, 3, 360, 480)
            pad_img: (1, 3, 480, 520) - cur_img: (1, 3, 390, 520)
            lseg_outputs: (1, 5, 480(ph), 520(pw))
            pad_crop_img: (1, 3, 480, 480)
        '''
        with torch.cuda.device_of(image):
            with torch.no_grad():
                vlmaps_outputs = image.new().resize_(batch,model.out_c,ph,pw).zero_().cuda()    # embedding vector 크기에 맞춰서 
                lseg_outputs = image.new().resize_(batch,len(labels),ph,pw).zero_().cuda()      
            count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)

                crop_img = crop_image(pad_img, h0, h1, w0, w1)      # 1, 3, 480, 480
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean,
                                            norm_std, crop_size)
                with torch.no_grad():
                    # outputs = module_inference(self.module, pad_img, label_set, self.flip)
                    vlmaps_output, lseg_output = model(pad_crop_img, labels)
                vlmaps_cropped = crop_image(vlmaps_output, 0, h1-h0, 0, w1-w0)           # (1, 5, 480, 480)
                vlmaps_outputs[:,:,h0:h1,w0:w1] += vlmaps_cropped
                
                lseg_cropped = crop_image(lseg_output, 0, h1-h0, 0, w1-w0)           # (1, 5, 480, 480)
                lseg_outputs[:,:,h0:h1,w0:w1] += lseg_cropped

                count_norm[:,:,h0:h1,w0:w1] += 1
        assert((count_norm==0).sum()==0)
        vlmaps_outputs = vlmaps_outputs / count_norm
        vlmaps_outputs = vlmaps_outputs[:,:,:height,:width]
        
        lseg_outputs = lseg_outputs / count_norm
        lseg_outputs = lseg_outputs[:,:,:height,:width]     # (1, 5, 390, 520)
    vlmaps_outputs = vlmaps_outputs.cpu()
    vlmaps_outputs = vlmaps_outputs.numpy()
    
    lseg_score = resize_image(lseg_outputs, h, w, **{'mode': 'bilinear', 'align_corners': True})
    lseg_scores += lseg_score

    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in lseg_outputs]
    pred = predicts[0]

    '''
    (1, class_num, h(720), w(1080)) - torch.float32, (1, embedding_dim, 390, 520) - float32, (390, 520) - int64
    '''
    return lseg_scores, vlmaps_outputs, pred # mask image, image embedding (batch 포함)

# original code
def original_get_lseg_feat(model, img_path, labels, transform, crop_size=480, \
                 base_size=520, norm_mean=[0.5, 0.5, 0.5], norm_std=[0.5, 0.5, 0.5]):
    # Step0. load a image
    image = Image.open(img_path)    
    image = np.array(image)                                                     # 360, 480, 3       - H, W, D

    image = transform(image).unsqueeze(0).cuda()
    img = image[0].permute(1,2,0)
    img = img * 0.5 + 0.5

    batch, _, h, w = image.size()
    stride_rate = 2.0/3.0
    stride = int(crop_size * stride_rate)

    long_size = base_size
    if h > w:
        height = long_size
        width = int(1.0 * w * long_size / h + 0.5)
        short_size = width
    else:
        width = long_size
        height = int(1.0 * h * long_size / w + 0.5)
        short_size = height


    cur_img = resize_image(image, height, width, **{'mode': 'bilinear', 'align_corners': True})

    if long_size <= crop_size:
        pad_img = pad_image(cur_img, norm_mean,
                            norm_std, crop_size)
        print(pad_img.shape)
        with torch.no_grad():
            outputs, logits = model(pad_img, labels)
        outputs = crop_image(outputs, 0, height, 0, width)
    else:
        if short_size < crop_size:
            # pad if needed
            pad_img = pad_image(cur_img, norm_mean,
                                norm_std, crop_size)
        else:
            pad_img = cur_img
        _,_,ph,pw = pad_img.shape #.size()
        assert(ph >= height and pw >= width)
        h_grids = int(math.ceil(1.0 * (ph-crop_size)/stride)) + 1
        w_grids = int(math.ceil(1.0 * (pw-crop_size)/stride)) + 1
        with torch.cuda.device_of(image):
            with torch.no_grad():
                outputs = image.new().resize_(batch, model.out_c,ph,pw).zero_().cuda()
                logits_outputs = image.new().resize_(batch, len(labels),ph,pw).zero_().cuda()
            count_norm = image.new().resize_(batch,1,ph,pw).zero_().cuda()
        # grid evaluation
        for idh in range(h_grids):
            for idw in range(w_grids):
                h0 = idh * stride
                w0 = idw * stride
                h1 = min(h0 + crop_size, ph)
                w1 = min(w0 + crop_size, pw)
                crop_img = crop_image(pad_img, h0, h1, w0, w1)
                # pad if needed
                pad_crop_img = pad_image(crop_img, norm_mean,
                                            norm_std, crop_size)
                with torch.no_grad():
                    output, logits = model(pad_crop_img, labels)
                cropped = crop_image(output, 0, h1-h0, 0, w1-w0)
                cropped_logits = crop_image(logits, 0, h1-h0, 0, w1-w0)
                outputs[:,:,h0:h1,w0:w1] += cropped
                logits_outputs[:,:,h0:h1,w0:w1] += cropped_logits
                count_norm[:,:,h0:h1,w0:w1] += 1
        assert((count_norm==0).sum()==0)
        outputs = outputs / count_norm
        logits_outputs = logits_outputs / count_norm
        outputs = outputs[:,:,:height,:width]
        logits_outputs = logits_outputs[:,:,:height,:width]
    outputs = outputs.cpu()
    outputs = outputs.numpy() # B, D, H, W
    
    predicts = [torch.max(logit, 0)[1].cpu().numpy() for logit in logits_outputs]
    pred = predicts[0]

    return outputs                                  # (1, 512, 390, 520)                                                  