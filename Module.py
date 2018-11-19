import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet34

def IOU(Reframe,GTframe):
    """
    自定义函数，计算两矩形 IOU，传入为均为矩形对角线，（x,y）  坐标。
    """
    x1 = Reframe[0]
    y1 = Reframe[1]
    width1 = Reframe[2] - Reframe[0]
    height1 = Reframe[3] - Reframe[1]

    x2 = GTframe[0]
    y2 = GTframe[1]
    width2 = GTframe[2] - GTframe[0]
    height2 = GTframe[3] - GTframe[1]

    endx = max(x1 + width1, x2 + width2)
    startx = min(x1, x2)
    width = width1 + width2 - (endx - startx)

    endy = max(y1 + height1, y2 + height2)
    starty = min(y1, y2)
    height = height1 + height2 - (endy - starty)

    if width <=0 or height <= 0:
        ratio = 0 # 重叠率为 0 
    else:
        Area = width * height # 两矩形相交面积
        Area1 = width1 * height1
        Area2 = width2 * height2
        ratio = Area * 1. / (Area1 + Area2 - Area)
    # return IOU
    return ratio, Reframe, GTframe

def CAM(feat_conv, weight_softmax, class_idx):

    bz, nc, h, w = feat_conv.shape
    output_cam = []
    for i, idx in enumerate(class_idx):
        cam = weight_softmax[idx].dot(feat_conv[i].reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        output_cam.append(cam_img)
    return output_cam

class RegionProposal(nn.Module):

    def __init__(self, anchors):
        super(RegionProposal, self).__init__()

        self.RPN_Conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.RPN_cls_score = nn.Conv2d(512, 10, 1, 1)
        self.RPN_bbox_pred = nn.Conv2d(512, 20, 1, 1)

        self.RPN_anchor_target = AnchorTargetLayer(anchors)
        self.RPN_proposal = ProposalLayer()
    
    def forward(self, x, bbox=None):
        x = self.RPN_Conv(x)
        cls_score = self.RPN_cls_score(x)  ## cls_score: [batch, 2k, h, w]
        bbox_pred = self.RPN_bbox_pred(x)  ## bbox_pred: [batch, 4k, h, w]

        if bbox is not None:  ## training stage
            '''
            utilize RPN_anchor_target to generate fg/bg info
            and bbox regression
            In output of RPN_anchor_target, there are anchor_labels,
            target_bbox and isValid, which indicate whether the anchor is 
            background or not, the fittest bbox to which the anchor belong and 
            whether the anchor is put into traning
            '''
            output = self.RPN_anchor_target(x, bbox)
            


class ProposalLayer(nn.Module):
    """
    Output proposed regions with the 
    topN scores
    """
    def __init__(self, topN):
        super(ProposalLayer, self).__init__()
        self.topN = topN
    
    def forward(self, inputs):
        '''

        '''
        scores = inputs[0][:, 5:, :, :] ## scores: [batch, 5, height, width]
        bbox_delta = inputs[1] ## bbox_delta: [batch, height, width, 4]

        feat_height, feat_width = scores.size(2), scores.size(3)
        batch_size = bbox_delta.size(0)
        proposals = torch.zeros()

        for i in range(batch_size):
            score = scores[i]
            delta = bbox_delta[i]
            info = feat_info[i]
            
            score_reshape = score.view(-1)
            order = torch.sort(score_reshape, descending=True)[:self.topN]
            for j in range(self.topN):
                idx = 0
                h = order[j] / feat_height
                w = order[j] % feat_width
                proposals = torch.cat([proposals, bbox_transform(idx, h, w, delta)])

        return proposals
        

class AnchorTargetLayer(nn.Module):
    '''
    assign each anchor to a bounding box
    '''
    def __init__(self, anchors):
        super(AnchorTargetLayer, self).__init__()
        self.anchors = anchors
    
    def forward(self, feat_map, bbox):
        feat_height, feat_width = feat_map.size(3),feat_map.size(4)
        batch_size = feat_map.size(0)
        cls_label = []
        target_bbox = []
        isValid = []


        for i in range(batch_size):
            feat = feats[i]   ## feat: [batch, height, width]
            for h in range(1, feat_width - 1):
                for w in range(1, feat_height - 1):
                '''
                each position will generate some anchors
                anchors store (dx, dy, h, w)
                if the position's coordinates are (x, y), the center
                of the anchor is (x+dx, y+dy). Then use (h, w) to
                get the both left top and right down of the anchor 
                '''
                    for anchor in self.anchors:
                        cx, cy = h + anchor[0], w + anchor[1]
                        lx, ly = cx - 0.5 * anchor[2], cy - 0.5 * anchor[3]
                        rx, ry = cx + 0.5 * anchor[2], cy + 0.5 * anchor[3]
                        
                        for j in range(len(bbox)):
                            b_lx, b_ly = bbox[j, 0] - 0.5 * bbox[j, 2], bbox[j, 1] - 0.5 * bbox[j, 3]
                            b_rx, b_ry = bbox[j, 0] + 0.5 * bbox[j, 2], bbox[j, 1] + 0.5 * bbox[j, 3]

                            iou = IOU(Reframe=(lx, ly, rx, ry), GTframe=(b_lx, b_ly, b_rx, b_ry))
                            if iou > 0.5:
                                cls_label.append([0, 1])
                                isValid.append(1)
                            else if iou < 0.3:
                                cls_label.append([1, 0])
                                isValid.append(0)
                            else:
                                cls_label.append([0, 0])
                                isValid.append(0)
                            target_bbox.append([bbox[j, 0], bbox[j, 1], bbox[j, 2], bbox[j, 3]])

        output = [cls_label, target_bbox, isValid]
        return output

class FeatMap(nn.Module):

    def __init__(self, num_classes):

        super(FeatMap, self).__init__()
        self.num_classes = num_classes
        resnet = resnet34(pretrained=True)
        self.pre = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.fc = nn.Linear(512, num_classes, bias=False)
    
    def forward(self, img):
        output = self.pre(img)
        output = self.layer(output)
        return output
