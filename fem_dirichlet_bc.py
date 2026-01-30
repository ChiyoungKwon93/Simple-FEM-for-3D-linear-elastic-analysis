# 모듈 import
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

# ----------------------------
# 3) Dirichlet BC(u=v=w=0) 적용
#    (Kg에 대해 Dirichlet BC dofs 대응되는 행/열 0 만들고 대각 성분만 1 세팅)
# ----------------------------
# Kg는 지금 lil_matrix로 이미 만들었으니 그대로 수정 가능.
# 혹시 csr이면 lil로 바꾸고 진행:
def dirichlet_bc(Kg, F, disp_nodes):
    ndof = Kg.shape[0]

    if not isinstance(Kg, lil_matrix):
        Kg = Kg.tolil()

    # Dirichlet BC 적용할 dof 리스트 생성
    fixed_dofs = []
    for tag_node in disp_nodes:
        base = (tag_node - 1) * 3
        fixed_dofs.extend([base + 0, base + 1, base + 2])  # u,v,w = 0

    fixed_dofs = np.array(fixed_dofs, dtype=int)

    # ----------------------------------------- Dirichlet BC 적용: 행과 열을 0으로 만들고 대각 성분을 1로 세팅
    # 먼저 행(row)부터 처리
    # u,v,w=0이므로 RHS 조정은 필요 없음(일반값이면 F -= K[:,dof]*u0 해야 함)
    for dof in fixed_dofs:
        Kg.rows[dof] = [dof] # dof 번째 행과 연결된 열 중에서 dof 번째 열만 남기고 나머지 열은 제거(즉 대각 성분만 남김)
        Kg.data[dof] = [1.0] # dof번째 대각 성분을 1로 설정
        F[dof] = 0.0 # RHS도 0으로 설정(부여된 변위가 0이므로)

    # 열(column)도 0으로 만들어 symmetry 유지
    # (LIL에서 열 제거는 직접 루프가 필요)
    fixed_set = set(fixed_dofs.tolist())
    for i in range(ndof):
        if i in fixed_set: # 이미 대각 성분만 남긴 행은 건너뜀
            continue
        
        # 아직 처리 안 한 행에 대해 열 제거 진행
        row_cols = Kg.rows[i] # 아직 처리 안 한 i번째 행의 열 인덱스 리스트
        row_data = Kg.data[i] # 아직 처리 안 한 i번째 행의 데이터 리스트
        
        # fixed_dofs에 해당하는 열 성분 제거
        new_cols = []
        new_data = []
        for col, value in zip(row_cols, row_data):
            if col in fixed_set:
                continue
            new_cols.append(col)
            new_data.append(value)
        Kg.rows[i] = new_cols # i번째 행의 열을 fixed_dofs 제거한 새 열 리스트로 교체
        Kg.data[i] = new_data # i번째 행의 데이터를 fixed_dofs 제거한 새 데이터 리스트로 교체

    # solver용 CSR 변환
    K_csr = Kg.tocsr()

    return K_csr, F