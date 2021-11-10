import numpy as np
import pytorch_lightning as pl
import rasterio
import segmentation_models_pytorch as smp
import torch


class FloodModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name='timm-efficientnet-l2',
            encoder_weights=None,
            decoder_attention_type='scse',
            in_channels=9,
            classes=2,
        ).cuda()

    def forward(self, image):
        # Forward pass
        return self.model(image)
    
    def norm(self, arr, mn, mx):
        return (np.clip(arr, mn, mx) - mn) / (mx - mn)

    def std(self, arr, mean, std):
        return ( arr - mean) / std

    def predict(self, vv_path, vh_path, change_path, extent_path, occur_path, recurr_path, seas_path, trans_path, nasadem_path):
        # Switch on evaluation mode
        self.model.eval()
        torch.set_grad_enabled(False)

        # Create a 2-channel image
        with rasterio.open(vv_path) as vv:
            vv_img = vv.read(1)
        with rasterio.open(vh_path) as vh:
            vh_img = vh.read(1)

        with rasterio.open(nasadem_path) as vh:
            elev = vh.read(1)
        with rasterio.open(extent_path) as vh:
            extent = vh.read(1)
        with rasterio.open(occur_path) as vh:
            occur = vh.read(1)
        with rasterio.open(recurr_path) as vh:
            recurr = vh.read(1)
        with rasterio.open(seas_path) as vh:
            seas = vh.read(1)
        with rasterio.open(trans_path) as vh:
            trans = vh.read(1)
        with rasterio.open(change_path) as vh:
            change = vh.read(1)
        
#         # Min-max normalization for original chips
#         min_norm = -77
#         max_norm = 26

#         vv_img = self.norm(vv_img, min_norm, max_norm)
#         vh_img = self.norm(vh_img, min_norm, max_norm)
#         elev_min = -64
#         elev_max = 2091
#         extent_min = occur_min = recurr_min = seas_min = change_min = 0
#         extent_max = occur_max = recurr_max = seas_max = change_max = 255
#         trans_min = 0
#         trans_max = 10
        
#         elev = self.norm(elev, elev_min, elev_max)
#         extent = self.norm(extent, extent_min, extent_max)
#         occur = self.norm(occur, occur_min, occur_max)
#         recurr = self.norm(recurr, recurr_min, recurr_max)
#         seas = self.norm(seas, seas_min, seas_max)
#         trans = self.norm(trans, trans_min, trans_max)
#         change = self.norm(change, change_min, change_max)
        
        # mean std standardization
        vh_mean = -17.547868728637695
        vh_std = 5.648269176483154
        vv_mean = -10.775313377380371
        vv_std = 5.038400650024414
        elev_mean = 151.06439787523334
        elev_std = 147.7373788520009
        extent_mean = 3.4183792170563305
        extent_std = 28.79923820408129
        occur_mean = 8.270102377747257
        occur_std = 34.48069374950159
        recurr_mean = 12.422072491522645
        recurr_std = 38.354506653571214
        seas_mean = 3.918482361684426
        seas_std = 28.87351299843167
        change_mean = 236.96356944404405
        change_std = 47.62208238871771
        trans_mean = 0.5896631853166981
        trans_std = 1.9514194857824525
        
        vv_img = self.std(vv_img, vv_mean, vv_std)
        vh_img = self.std(vh_img, vh_mean, vh_std)
        elev   = self.std(elev, elev_mean, elev_std)
        extent = self.std(extent, extent_mean, extent_std)
        occur  = self.std(occur, occur_mean, occur_std)
        recurr = self.std(recurr, recurr_mean, recurr_std)
        seas   = self.std(seas, seas_mean, seas_std)
        trans  = self.std(trans, trans_mean, trans_std)
        change = self.std(change, change_mean, change_std)

            
        x_arr = np.stack([
                            vv_img,
                            vh_img,
                            elev,
                            extent,
                            occur, 
                            recurr, 
                            seas, 
                            trans, 
                            change, 
                          ], axis=-1
                        ).astype(np.float32)

        # Transpose
        x_arr = np.transpose(x_arr, [2, 0, 1])
        x_arr = np.expand_dims(x_arr, axis=0)

        # Perform inference
        x = torch.from_numpy(x_arr)
        x = x.cuda(non_blocking=True)

        preds = self.forward(x)
        preds = torch.softmax(preds, dim=1)[:, 1]
        preds = (preds > 0.5) * 1
        return preds.detach().cpu().numpy().astype(np.uint8).squeeze().squeeze()
