import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn


class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None,
                 dice_class=SoftDiceLoss):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = target != self.ignore_label
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.where(mask, target, 0)
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0]) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, weight_ce=1, weight_dice=1, use_ignore_label: bool = False,
                 dice_class=MemoryEfficientSoftDiceLoss):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        target mut be one hot encoded
        IMPORTANT: We assume use_ignore_label is located in target[:, -1]!!!

        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()
        if use_ignore_label:
            bce_kwargs['reduction'] = 'none'

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.use_ignore_label = use_ignore_label

        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = dice_class(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.use_ignore_label:
            # target is one hot encoded here. invert it so that it is True wherever we can compute the loss
            if target.dtype == torch.bool:
                mask = ~target[:, -1:]
            else:
                mask = (1 - target[:, -1:]).bool()
            # remove ignore channel now that we have the mask
            # why did we use clone in the past? Should have documented that...
            # target_regions = torch.clone(target[:, :-1])
            target_regions = target[:, :-1]
        else:
            target_regions = target
            mask = None

        dc_loss = self.dc(net_output, target_regions, loss_mask=mask)
        target_regions = target_regions.float()
        if mask is not None:
            ce_loss = (self.ce(net_output, target_regions) * mask).sum() / torch.clip(mask.sum(), min=1e-8)
        else:
            ce_loss = self.ce(net_output, target_regions)
        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super().__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables ' \
                                         '(DC_and_CE_loss)'
            mask = (target != self.ignore_label).bool()
            # remove ignore label from target, replace with one of the known labels. It doesn't matter because we
            # ignore gradients in those areas anyway
            target_dice = torch.clone(target)
            target_dice[target == self.ignore_label] = 0
            num_fg = mask.sum()
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) \
            if self.weight_dice != 0 else 0
        ce_loss = self.ce(net_output, target) \
            if self.weight_ce != 0 and (self.ignore_label is None or num_fg > 0) else 0

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        return result

class KD_MSE_loss(nn.Module):
    def __init__(self, reduction = 'mean', softmax = False, temperature = 2):
        super(KD_MSE_loss, self).__init__()
        
        self.reduction = reduction
        self.softmax = softmax
        self.temperature = temperature
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, net_output: torch.Tensor, target: torch.Tensor):

        softmax = self.softmax
        T = self.temperature

        teacher_output = target
        student_output = net_output
        
        if softmax:
            teacher_output = nn.functional.softmax(target/T, dim=1)
            student_output = nn.functional.softmax(net_output/T, dim=1)
            result = self.mse_loss(student_output, teacher_output) * (T**2)

        else:
            result = self.mse_loss(student_output/T, teacher_output/T)
        
        # Check the loss
        return result

class KD_KLDiv_loss(nn.Module):
    def __init__(self, reduction  = 'mean', temperature = 2):
        super(KD_KLDiv_loss, self).__init__()
        
        self.reduction = reduction
        self.temperature = temperature
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        T = self.temperature

        teacher_output = nn.functional.log_softmax(target/T, dim=1)
        student_output = nn.functional.log_softmax(net_output/T, dim=1)
        
        result = self.kldiv_loss(student_output, teacher_output) * (T**2)

        # Check the loss
        return result
    
class KD_KLDiv_loss(nn.Module):
    def __init__(self, reduction  = 'mean', temperature = 2):
        super(KD_KLDiv_loss, self).__init__()
        
        self.reduction = reduction
        self.temperature = temperature
        self.kldiv_loss = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        T = self.temperature
        teacher_output = nn.functional.log_softmax(target/T, dim=1)
        student_output = nn.functional.log_softmax(net_output/T, dim=1)
        
        result = self.kldiv_loss(student_output, teacher_output) * (T**2)

        # Check the loss
        return result
    
class KD_CE_loss(nn.Module):
    def __init__(self, reduction = 'mean', temperature = 2):
        super(KD_CE_loss, self).__init__()
        
        self.reduction = reduction
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        T = self.temperature

        teacher_output = nn.functional.softmax(target/T, dim=1)
        student_output = net_output/T
        
        result = self.ce_loss(student_output, teacher_output) * (T**2)

        # Check the loss
        return result


# Legacy
class Area_MSE_loss(nn.Module):
    def __init__(self, reduction = 'sum'):
        super(Area_MSE_loss, self).__init__()
        
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)
        
    def forward(self, net_output: torch.Tensor, target: torch.Tensor, ignore_area: torch.Tensor):
        total_voxels = ignore_area.numel()
        zero_voxels = (ignore_area==0).sum().item()
        area_Num = total_voxels - zero_voxels

        result = self.mse_loss(net_output*ignore_area, target*ignore_area) / area_Num

        return result

class Area_KLDiv_loss(nn.Module):
    def __init__(self):
        super(Area_KLDiv_loss, self).__init__()
        
        self.kldiv_loss = nn.KLDivLoss(reduction='mean', log_target=True)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, ignore_area: torch.Tensor):
        # total_voxels = ignore_area.size(0)  # 배치 크기 B
        # zero_voxels = (ignore_area == 0).sum(dim=(1, 2, 3, 4))  # 각 배치별 0인 요소 개수
        # area_Num = (ignore_area.numel() - zero_voxels.sum().item()) / total_voxels  # 배치별 평균 영역

        # Softmax
        # target = teacher / net_output = student
        temperature = 5
        # log_softmax에서 dim=1 옵션 제거
        teacher_output_softmax = nn.functional.log_softmax(target/temperature, dim=1)
        student_output_softmax = nn.functional.log_softmax(net_output/temperature, dim=1)
        # masked_student = student_output_softmax*ignore_area
        # masked_teacher  = teacher_output_softmax*ignore_area

        # Debugging information
        # print("Teacher output softmax:")
        # print(f"  Shape: {teacher_output_softmax.shape}")
        # print(f"  Min: {teacher_output_softmax.min().item()}, Max: {teacher_output_softmax.max().item()}, Mean: {teacher_output_softmax.mean().item()}")

        # print("Student output softmax:")
        # print(f"  Shape: {student_output_softmax.shape}")
        # print(f"  Min: {student_output_softmax.min().item()}, Max: {student_output_softmax.max().item()}, Mean: {student_output_softmax.mean().item()}")


        result = self.kldiv_loss(teacher_output_softmax, student_output_softmax)

        return result

class Area_KLDiv_loss2(nn.Module):
    def __init__(self):
        super(Area_KLDiv_loss2, self).__init__()
        
        self.kldiv_loss = nn.KLDivLoss(reduction='mean', log_target=True)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, ignore_area: torch.Tensor):
        # total_voxels = ignore_area.size(0)  # 배치 크기 B
        # zero_voxels = (ignore_area == 0).sum(dim=(1, 2, 3, 4))  # 각 배치별 0인 요소 개수
        # area_Num = (ignore_area.numel() - zero_voxels.sum().item()) / total_voxels  # 배치별 평균 영역

        # Softmax
        # target = teacher / net_output = student
        temperature = 10
        # log_softmax에서 dim=1 옵션 제거
        teacher_output_softmax = nn.functional.log_softmax(target/temperature, dim=1)
        student_output_softmax = nn.functional.log_softmax(net_output/temperature, dim=1)
        # masked_student = student_output_softmax*ignore_area
        # masked_teacher  = teacher_output_softmax*ignore_area

        # Debugging information
        # print("Teacher output softmax:")
        # print(f"  Shape: {teacher_output_softmax.shape}")
        # print(f"  Min: {teacher_output_softmax.min().item()}, Max: {teacher_output_softmax.max().item()}, Mean: {teacher_output_softmax.mean().item()}")

        # print("Student output softmax:")
        # print(f"  Shape: {student_output_softmax.shape}")
        # print(f"  Min: {student_output_softmax.min().item()}, Max: {student_output_softmax.max().item()}, Mean: {student_output_softmax.mean().item()}")


        result = self.kldiv_loss(teacher_output_softmax, student_output_softmax)

        return result