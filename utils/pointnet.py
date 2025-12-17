import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D >3 :
            x, feature = x.split(3,dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        # if self.global_feat:
        #     return x, trans, trans_feat
        # else:
        #     x = x.view(-1, 1024, 1).repeat(1, 1, N)
        #     return torch.cat([x, pointfeat], 1), trans, trans_feat
        return x


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss

class PointNetGravityBaseline(nn.Module):
    def __init__(self, channel=3):
        super(PointNetGravityBaseline, self).__init__()
        
        # 1. Standard PointNet Encoder (사용자님 코드)
        # global_feat=True로 해서 (B, 1024)를 뽑아냄
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        
        # 2. Regression Head (Standard MLP)
        # VNLinear가 아닌 일반 nn.Linear 사용 -> 회전 등변성 없음(Invariant)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3) # 최종 3D 벡터 (Gravity)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        # Encoder: (B, 3, N) -> (B, 1024)
        x = self.feat(x)
        
        # MLP Head
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x) # (B, 3)
        
        # 중력 벡터이므로 Unit Vector로 정규화
        x = F.normalize(x, p=2, dim=1)
        
        return x
    
if __name__ == '__main__':
    
    # 1. 데이터 생성 (Batch=10, Channel=3, Points=30)
    B, C, N = 10, 3, 30 
    points = torch.randn(B, C, N)
    print("Input Points Shape:", points.shape)
    
    # 2. 회전 행렬 생성 (45도)
    theta = np.pi / 4 
    rotation_matrix = torch.tensor([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ], dtype=torch.float32)
    
    # 입력 데이터 회전
    rotated_points = torch.matmul(rotation_matrix, points) 
    print("Rotated Points Shape:", rotated_points.shape)
    
    # -------------------------------------------------------------------------
    # [핵심 수정] PointNetEncoder 대신 PointNetGravityBaseline 사용!
    # -------------------------------------------------------------------------
    # 이유: 회전 등변성을 테스트하려면 출력이 3차원 벡터여야 함 (1024차원은 회전 불가)
    model = PointNetGravityBaseline(channel=3)
    
    # [중요] 평가 모드 (BatchNorm 오류 방지 및 결과 고정)
    model.eval()
    
    # 3. 추론
    with torch.no_grad():
        pred1 = model(points)          # 원본 입력 -> 예측값 (B, 3)
        pred2 = model(rotated_points)  # 회전된 입력 -> 예측값 (B, 3)
    
    print("Output Shape:", pred1.shape) # (10, 3) 확인
    
    # 4. Equivariance 검증 (Non-Equivariant 확인)
    # f(Rx) vs R * f(x)
    
    # pred1(원본 예측값)을 수동으로 회전
    # (B, 3) @ (3, 3).T
    pred1_rotated = pred1 @ rotation_matrix.T  
    
    # 두 값의 차이 계산
    loss = F.mse_loss(pred2, pred1_rotated)
    
    print("\n=== [Baseline] Standard PointNet Equivariance Test ===")
    print(f"Equivariance Loss (MSE): {loss.item():.6f}")
    
    # 결과 해석 메시지 출력
    if loss.item() > 1e-3:
        print("\n✅ 확인 완료: Loss가 높습니다.")
        print("   -> 일반 PointNet은 입력이 회전해도 출력이 따라 돌지 못합니다 (Non-Equivariant).")
        print("   -> 반면, VN 모델은 이 Loss가 0.0에 가깝습니다.")
    else:
        print("\n❓ 이상 현상: Loss가 낮습니다. (우연히 맞았거나 초기화 영향)")