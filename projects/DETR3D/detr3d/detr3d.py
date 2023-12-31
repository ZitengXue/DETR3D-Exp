from typing import Dict, List, Optional
from torch.nn.init import normal_
import torch
from torch import Tensor
import torch.nn as nn
from typing import Dict, Tuple, Union
from .glip import (create_positive_map, create_positive_map_label_to_token,
                   run_ner)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d.utils import get_lidar2img
from .grid_mask import GridMask
from mmdet.structures import OptSampleList
import torch.nn.functional as F
from ..layers.transformer.grounding_dino_layers import (
    GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder)
from mmdet.models.layers.positional_encoding import SinePositionalEncoding

@MODELS.register_module()
class DETR3D(MVXTwoStageDetector):
    """DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries

    Args:
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`Det3DDataPreprocessor`. Defaults to None.
        use_grid_mask (bool) : Data augmentation. Whether to mask out some
            grids during extract_img_feat. Defaults to False.
        img_backbone (dict, optional): Backbone of extracting
            images feature. Defaults to None.
        img_neck (dict, optional): Neck of extracting
            image features. Defaults to None.
        pts_bbox_head (dict, optional): Bboxes head of
            detr3d. Defaults to None.
        train_cfg (dict, optional): Train config of model.
            Defaults to None.
        test_cfg (dict, optional): Train config of model.
            Defaults to None.
        init_cfg (dict, optional): Initialize config of
            model. Defaults to None.
    """

    def __init__(self,
                 data_preprocessor=None,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 language_model=None,
                 encoder=None,
                 positional_encoding_single=None,
                 ):
        super(DETR3D, self).__init__(
            img_backbone=img_backbone,
            img_neck=img_neck,
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.language_cfg = language_model
        self.language_model = MODELS.build(self.language_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            256,
            bias=True)
        self._special_tokens = '. '
        self.positional_encoding = SinePositionalEncoding(
            **positional_encoding_single)
        self.encoder = GroundingDinoTransformerEncoder(**encoder)
        # self.level_embed = nn.Parameter(
        #     torch.Tensor(4, 256))
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        # normal_(self.level_embed)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)
    def extract_img_feat(self, img: Tensor,
                         batch_input_metas: List[dict]) -> List[Tensor]:
        """Extract features from images.

        Args:
            img (tensor): Batched multi-view image tensor with
                shape (B, N, C, H, W).
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
             list[tensor]: multi-level image features.
        """

        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]  # bs nchw
            # update real input shape of each single img
            for img_meta in batch_input_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)  # mask out some grids
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    def extract_feat(self, batch_inputs_dict: Dict,
                     batch_input_metas: List[dict]) -> List[Tensor]:
        """Extract features from images.

        Refer to self.extract_img_feat()
        """
        imgs = batch_inputs_dict.get('imgs', None)
        img_feats = self.extract_img_feat(imgs, batch_input_metas)
        return img_feats

    def _forward(self):
        raise NotImplementedError('tensor mode is yet to add')

    # original forward_train
    def loss(self, batch_inputs_dict: Dict[List, Tensor],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        """
        Args:
            batch_inputs_dict (dict): The model input dict which include
                `imgs` keys.
                - imgs (torch.Tensor): Tensor of batched multi-view  images.
                    It has shape (B, N, C, H ,W)
            batch_data_samples (List[obj:`Det3DDataSample`]): The Data Samples
                It usually includes information such as `gt_instance_3d`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.

        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        bsz=len(batch_data_samples)
        #文本预处理
        text_prompts=[
        'car', 'truck', 'trailer', 'bus', 'construction vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic cone', 'barrier']
        batch_gt_instances_3d = [
            item.gt_instances_3d for item in batch_data_samples
        ]
        new_text_prompts=[]
        positive_maps=[]
        tokenized, caption_string, tokens_positive, _ = \
                self.get_tokens_and_prompts(
                    text_prompts, True)
        new_text_prompts = [caption_string] * len(batch_data_samples) 
        gt_labels=[
                data_sample.labels_3d 
                for data_sample in batch_gt_instances_3d
                ]
        for gt_label in gt_labels:
            new_tokens_positive = [
                    tokens_positive[label] for label in gt_label
                ]
            _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive)
            positive_maps.append(positive_map)

        text_dict = self.language_model(new_text_prompts)
        for key, value in text_dict.items():
            text_dict[key] = torch.cat([value] * 6, dim=0)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        #####################################################################
        encoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        memory = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        del img_feats
        img_feats = self.restore_img_feats(memory, encoder_inputs_dict['spatial_shapes'], encoder_inputs_dict['level_start_index'])
        outs = self.pts_bbox_head(img_feats, batch_input_metas, **kwargs)#text_dict
        loss_inputs = [batch_gt_instances_3d, outs]
        losses_pts = self.pts_bbox_head.loss_by_feat(*loss_inputs)

        return losses_pts

    # original simple_test
    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `imgs` keys.

                - imgs (torch.Tensor): Tensor of batched multi-view images.
                    It has shape (B, N, C, H ,W)
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 9).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        batch_input_metas = self.add_lidar2img(batch_input_metas)
        img_feats = self.extract_feat(batch_inputs_dict, batch_input_metas)
        bsz=len(batch_data_samples)
        #文本预处理
        text_prompts=[
        'car', 'truck', 'trailer', 'bus', 'construction vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic cone', 'barrier']

        new_text_prompts=[]
        positive_maps=[]
        tokenized, caption_string, tokens_positive, _ = \
                self.get_tokens_and_prompts(
                    text_prompts, True)
        new_text_prompts = [caption_string] * len(batch_data_samples) 
        text_dict = self.language_model(new_text_prompts)
        for key, value in text_dict.items():
            text_dict[key] = torch.cat([value] * 6, dim=0)
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])
        #####################################################################
        encoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        memory = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)
        del img_feats
        img_feats = self.restore_img_feats(memory, encoder_inputs_dict['spatial_shapes'], encoder_inputs_dict['level_start_index'])
        outs = self.pts_bbox_head(img_feats, batch_input_metas)

        results_list_3d = self.pts_bbox_head.predict_by_feat(
            outs, batch_input_metas, **kwargs)

        # change the bboxes' format
        detsamples = self.add_pred_to_datasample(batch_data_samples,
                                                 results_list_3d)
        return detsamples

    # may need speed-up
    def add_lidar2img(self, batch_input_metas: List[Dict]) -> List[Dict]:
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.

        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for meta in batch_input_metas:
            l2i = list()
            for i in range(len(meta['cam2img'])):
                c2i = torch.tensor(meta['cam2img'][i]).double()
                l2c = torch.tensor(meta['lidar2cam'][i]).double()
                l2i.append(get_lidar2img(c2i, l2c).float().numpy())
            meta['lidar2img'] = l2i
        return batch_input_metas
        
    def get_tokens_and_prompts(
            self,
            original_caption: Union[str, list, tuple],
            custom_entities: bool = False) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            caption_string = ''
            tokens_positive = []
            for idx, word in enumerate(original_caption):
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
                caption_string += self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities
    

    def get_tokens_and_prompts(
            self,
            original_caption: Union[str, list, tuple],
            custom_entities: bool = False) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            caption_string = ''
            tokens_positive = []
            for idx, word in enumerate(original_caption):
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
                caption_string += self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities
    
    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(tokenized, tokens_positive)
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map
    
    def pre_transformer(
            self,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = mlvl_feats[0].size(0)
        num_cams=mlvl_feats[0].size(1)
        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(
                    self.positional_encoding(None, input=feat))
        else:
            masks = mlvl_feats[0].new_ones(
                (batch_size, num_cams,input_img_h, input_img_w))
            for img_id in range(batch_size):
                for cam in range(num_cams):
                    img_h, img_w = img_shape_list[img_id][cam]
                    masks[img_id,cam, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks, size=feat.shape[-2:]).to(
                        torch.bool))
                tmp=[]
                for i in range(batch_size):
                    tmp.append(self.positional_encoding(mlvl_masks[-1][0]).unsqueeze(0))
                concatenated = torch.cat(tmp, dim=0)
                mlvl_pos_embeds.append(concatenated)

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, n,c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[3:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size,n,c, -1).permute(0, 1, 3, 2)
            pos_embed = pos_embed.view(batch_size,n, c, -1).permute(0, 1, 3, 2)
            # lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(2)

            feat_flatten.append(feat)
            # lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 2)
        # lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 2)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 2)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            tmp=[]
            for i in range(batch_size):
                for m in mlvl_masks:
                    tmp.append(self.get_valid_ratio(m[i]))
            valid_ratios = torch.stack(tmp,1).view(batch_size,num_cams,4,2)
        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            # feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)

        return encoder_inputs_dict
    
    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                         spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, _ = self.encoder(
            query=feat,
            # query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        # encoder_outputs_dict = dict(
        #     memory=memory,
        #     memory_mask=feat_mask,
        #     spatial_shapes=spatial_shapes,
        #     memory_text=memory_text,
        #     text_token_mask=text_token_mask)
        # return encoder_outputs_dict
        return memory
    @staticmethod
    def get_valid_ratio(mask: Tensor) -> Tensor:
        """Get the valid radios of feature map in a level.

        .. code:: text

                    |---> valid_W <---|
                 ---+-----------------+-----+---
                  A |                 |     | A
                  | |                 |     | |
                  | |                 |     | |
            valid_H |                 |     | |
                  | |                 |     | H
                  | |                 |     | |
                  V |                 |     | |
                 ---+-----------------+     | |
                    |                       | V
                    +-----------------------+---
                    |---------> W <---------|

          The valid_ratios are defined as:
                r_h = valid_H / H,  r_w = valid_W / W
          They are the factors to re-normalize the relative coordinates of the
          image to the relative coordinates of the current level feature map.

        Args:
            mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

        Returns:
            Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
        """
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    def restore_img_feats(self, memory, spatial_shapes, level_start_index):
        img_feats = []
        for i in range(len(spatial_shapes)):
            # 获取当前层次的空间尺寸
            spatial_shape = spatial_shapes[i]
            height, width = spatial_shape

            # 获取当前层次的起始索引和结束索引
            start_index = level_start_index[i]
            if i < len(level_start_index) - 1:
                end_index = level_start_index[i+1]
            else:
                end_index = memory.shape[2]

            # 切片操作，从memory中恢复当前层次的img_feat
            img_feat = memory[:, :, start_index:end_index]
            img_feat = img_feat.reshape(1, 6, height, width,256).permute(0,1,4,2,3)
            img_feats.append(img_feat)

        return img_feats