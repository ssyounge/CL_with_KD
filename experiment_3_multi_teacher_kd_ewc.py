import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser

# 1) FACIL의 ewc.py 코드를 상속
from .ewc import Appr as EWCApproach

########################################
# Teacher Manager (Multi-Teacher KD)
########################################
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_teachers, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_teachers)

    def forward(self, x):
        # x: [B, input_dim]
        h = F.relu(self.fc1(x))
        weights = F.softmax(self.fc2(h), dim=1)
        return weights

class TeacherManager(nn.Module):
    def __init__(self, teacher_ckpt_paths, create_teacher_fn, device,
                 gating_input_dim=16384, gating_hidden_dim=128):
        super().__init__()
        self.device = device
        self.num_teachers = len(teacher_ckpt_paths)

        # Teacher 모델 로딩
        self.teachers = nn.ModuleList()
        for ckpt in teacher_ckpt_paths:
            import os
            ckpt_abs = os.path.abspath(ckpt)
            
            teacher = create_teacher_fn()
            state = torch.load(ckpt_abs, map_location=device)
            teacher.load_state_dict(state)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.to(device)
            self.teachers.append(teacher)

        # Gating Net
        self.gating_net = GatingNetwork(gating_input_dim, self.num_teachers, gating_hidden_dim).to(device)

    def forward(self, x, student_feats):
        # x: [B, C, H, W] / student_feats: [B, gating_input_dim]
        with torch.no_grad():
            teacher_logits_list = []
            for tnet in self.teachers:
                logits = tnet(x)   # [B, num_classes]
                teacher_logits_list.append(logits)
            teacher_logits_stack = torch.stack(teacher_logits_list, dim=1)  # [B, T, num_classes]

        # Gating
        w = self.gating_net(student_feats)  # [B, T]
        w = w.unsqueeze(-1)                # [B, T, 1]
        ensemble_logits = torch.sum(teacher_logits_stack * w, dim=1)  # [B, num_classes]

        return ensemble_logits, w, teacher_logits_stack


########################################
# Multi-Teacher KD + EWC
########################################
class Appr(EWCApproach):
    """
    EWC (ewc.py)의 train_loop()를 오버라이드해 Multi-Teacher KD 로직을 추가.
    - 부모 __init__에 없는 키워드는 pop(...)하여 TypeError 방지
    - teacher checkpoint 로딩 시 resnet32(num_classes=100) 사용
    """

    def __init__(self, model, device, **kwargs):
        """
        1) 자식에서만 사용하는 인자를 pop
        2) 나머지를 부모 init에 전달
        3) TeacherManager 초기화
        """
        # 자식 측 인자 pop
        self.kd_weight = kwargs.pop('kd_weight', 1.0)
        self.kd_T = kwargs.pop('kd_temperature', 2.0)
        teacher_ckpts = kwargs.pop('teacher_checkpoints', '')
        gating_input_dim = kwargs.pop('gating_input_dim', 16384)
        gating_hidden_dim = kwargs.pop('gating_hidden_dim', 128)
        self.some_other_options = kwargs.pop('some_other_options', None)

        # 부모(EWCApproach) init: 나머지 kwargs만 넘김
        super().__init__(model, device, **kwargs)

        # Teacher checkpoint 로딩
        if teacher_ckpts.strip():
            ckpt_paths = [x.strip() for x in teacher_ckpts.split(',') if x.strip()]
            if len(ckpt_paths) > 0:
                def create_teacher():
                    # resnet32를 함수로 직접 호출
                    from networks.resnet32 import resnet32
                    net = resnet32(num_classes=100)  # CIFAR-100
                    return net

                self.teacher_manager = TeacherManager(
                    ckpt_paths, create_teacher, device,
                    gating_input_dim, gating_hidden_dim
                )

                # ─────────────────────────────────────────
                # **디버깅 프린트**: 여기서 gating_input_dim 실제 값과
                #                  fc1.weight.shape 등을 확인
                print(f"[DEBUG] gating_input_dim={gating_input_dim} in Appr __init__")
                print(f"[DEBUG] teacher_manager.gating_net.fc1.weight.shape="
                    f"{self.teacher_manager.gating_net.fc1.weight.shape}")
                # ─────────────────────────────────────────

            else:
                self.teacher_manager = None
        else:
            self.teacher_manager = None


    @staticmethod
    def extra_parser(args):
        """
        1) 먼저 부모 EWCApproach.extra_parser(args) → (ewc_ns, leftover)
        2) 자식 parser(child_parser) 생성하여 leftover를 parse
        3) 자식의 Namespace(child_ns)를 ewc_ns에 병합
        4) leftover2 반환
        """
        # 1) 부모 parser 결과
        ewc_ns, leftover = EWCApproach.extra_parser(args)

        # 2) 자식 parser
        import argparse
        child_parser = argparse.ArgumentParser(add_help=False)

        # 자식 측 인자들 (Multi-teacher KD 등)
        child_parser.add_argument('--kd-weight', type=float, default=1.0,
                                help='KD loss weight')
        child_parser.add_argument('--kd-temperature', type=float, default=2.0,
                                help='KD temperature')
        child_parser.add_argument('--teacher-checkpoints', type=str, default='',
                                help='Teacher 모델 ckpt 경로(쉼표 구분)')
        child_parser.add_argument('--gating-input-dim', type=int, default=4096,
                                help='ResNet32(n=5): [64,8,8]->4096')
        child_parser.add_argument('--gating-hidden-dim', type=int, default=128,
                                help='Gating network hidden size')
        # "some-other-options"도 leftover로 안 남게 등록
        child_parser.add_argument('--some-other-options', type=str, default=None,
                                help='Extra param for debugging')

        # 3) leftover parse
        child_ns, leftover2 = child_parser.parse_known_args(leftover)

        # 4) 병합: child_ns 항목 → ewc_ns에 복사
        for k, v in vars(child_ns).items():
            setattr(ewc_ns, k, v)

        return ewc_ns, leftover2


    ###################################################
    # EWCApproach에는 이미 train(), train_loop()가 있음.
    # 여기서 train_loop()만 오버라이드하되, 대부분 로직은 그대로 사용.
    ###################################################


    def train_loop(self, t, trn_loader, val_loader):
        self.optimizer = self._get_optimizer()

        # epoch-based log
        lr = self.lr
        best_loss = float('inf')
        patience = self.lr_patience
        best_model = None

        for e in range(self.nepochs):
            # 1) 한 epoch의 train
            avg_train_loss = self.train_epoch(t, trn_loader, epoch=e)

            # 2) validation
            valid_loss, valid_acc, _ = self.eval(t, val_loader)

            # (추가) logger에 epoch별 train_loss, valid_loss/acc 기록
            if hasattr(self, 'logger') and self.logger is not None:
                self.logger.log_scalar(task=t, iter=e,
                                       name="train_loss", value=avg_train_loss,
                                       group="train")
                self.logger.log_scalar(task=t, iter=e,
                                       name="valid_loss", value=valid_loss,
                                       group="valid")
                self.logger.log_scalar(task=t, iter=e,
                                       name="valid_acc",  value=valid_acc,
                                       group="valid")

            # 3) compare / patience / lr schedule
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = self.model.get_copy()
                patience = self.lr_patience
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    if lr < self.lr_min:
                        print("Early stopping as lr < lr_min")
                        break
                    patience = self.lr_patience
                    # restore best so far
                    self.model.set_state_dict(best_model)
                    # update optimizer lr
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr

        # after training => EWC post
        # 4) after training, do post train process for EWC
        self.post_train_process(t, trn_loader)


    def train_epoch(self, t, trn_loader, epoch=None):
        """
        (2) 한 epoch 내에서 mini-batch별 KD 로직 추가
        """
        if epoch is None:
            epoch = 0

        total_loss = 0.
        num_batches = 0

        self.model.train()
        for x, y in trn_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model.forward(x)

            # EWC + CE (부모 criterion)
            loss = super().criterion(t, outputs, y)

            # (2-1) KD 계산
            if self.teacher_manager and t>0:
                feats = self._extract_student_features(x)
                teacher_logits, _, _ = self.teacher_manager(x, feats)
                kd_loss = self._compute_kd_loss(outputs, teacher_logits)
                loss += kd_loss

                # (옵션) logger에 kd_loss 기록
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.log_scalar(task=t, iter=epoch, 
                                        name="kd_loss", value=kd_loss.item(),
                                        group="train")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        # 중요: 평균 loss를 반환
        return total_loss / max(1, num_batches)

    def _compute_kd_loss(self, outputs, teacher_logits):
        """
        KL div에 temperature 적용
        """
        # Student outputs -> shape e.g. [B, 40] or [B, 100], etc.
        student_out = torch.cat(outputs, dim=1)
        T = self.kd_T

        s_dim = student_out.size(1)     # e.g. 40
        t_dim = teacher_logits.size(1)  # e.g. 100

        # 1) Teacher > Student 인 경우 => Teacher logits을 잘라서 크기 맞춤
        if t_dim > s_dim:
            teacher_logits = teacher_logits[:, :s_dim]

        # 2) Teacher < Student 인 경우 => Student logits을 잘라서 크기 맞춤
        elif t_dim < s_dim:
            # 기존 코드엔 raise만 있었지만, 아래처럼 Student도 slice
            student_out = student_out[:, :t_dim]

        # 이제 두 tensor dimension이 동일
        # e.g. teacher_logits.size(1) == student_out.size(1)

        kd_term = F.kl_div(
            F.log_softmax(student_out / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T**2)

        return self.kd_weight * kd_term

    def _extract_student_features(self, x):
        with torch.no_grad():
            net = self.model.model
            # 1) conv1 → bn1 → relu
            out = net.relu(net.bn1(net.conv1(x)))
            # 2) layer1 → layer2 → layer3
            out = net.layer1(out)
            out = net.layer2(out)
            out = net.layer3(out)
            # 3) flatten
            out = out.view(out.size(0), -1)  # shape [B, 64*8*8=4096]
        return out
