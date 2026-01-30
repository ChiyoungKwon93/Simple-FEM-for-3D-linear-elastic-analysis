# global stiffness matrix sparse matrix로 생성하기 위한 모듈 import
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

### Element material properties matrix 생성 함수
# returns dN/dxi, dN/deta, dN/dzeta as a (3,8) array
def dNdxi_hex8(xi, eta, zeta):
    # Gmsh Hex8 node signs in the order [1..8] = [0..7]
    sxi  = np.array([-1, +1, +1, -1, -1, +1, +1, -1], dtype=float)
    seta = np.array([-1, -1, +1, +1, -1, -1, +1, +1], dtype=float)
    szeta = np.array([-1, -1, -1, -1, +1, +1, +1, +1], dtype=float)
    # shape function 미분값 계산
    dN_dxi   = 0.125 * sxi  * (1 + seta*eta) * (1 + szeta*zeta)
    dN_deta  = 0.125 * seta * (1 + sxi*xi)   * (1 + szeta*zeta)
    dN_dzeta = 0.125 * szeta * (1 + sxi*xi)   * (1 + seta*eta)
    return np.vstack([dN_dxi, dN_deta, dN_dzeta])  # (3,8)

# Element material properties matrix 계산 함수 : Hex8 요소 전용
def e_mat_prop_hex8(data_e:np.ndarray, ndarray_nodes:np.ndarray, C:np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    data_e : (10,) int ndarray
        elem_tag, elem_type, element connectivity (8 node indices)
    ndarray_nodes : (nnode, 4) float ndarray
        node_tag, Global nodal coordinates(x,y,z)
    C : (6, 6) float ndarray
        Material property matrix

    Returns
    -------
    Ke : (3*8, 3*8) float ndarray
        Element material properties matrix for Hex8
    """

    # Gauss 점과 가중치
    xi_g_points = [-0.5773502691896257, 0.5773502691896257]
    eta_g_points = [-0.5773502691896257, 0.5773502691896257]
    zeta_g_points = [-0.5773502691896257, 0.5773502691896257]
    w_g = [1.0, 1.0]
    
    # Element별 보유 node coordinates 추출
    node_coords = []
    for tag_node in data_e[2:10]:
        node_coords.append(ndarray_nodes[tag_node - 1, 1:])
    node_coords = np.array(node_coords)  # (8,3)

    # Element material properties matrix 초기화
    Ke = np.zeros((3*8, 3*8))

    # Gauss 점 하나씩 순회
    for i_g, xi_g in enumerate(xi_g_points):
        for j_g, eta_g in enumerate(eta_g_points):
            for k_g, zeta_g in enumerate(zeta_g_points):
                w = w_g[i_g] * w_g[j_g] * w_g[k_g]

                # shape function 미분값 계산
                dNdxi_g = dNdxi_hex8(xi_g, eta_g, zeta_g)

                # Jacobian 행렬 계산
                J_g = dNdxi_g @ node_coords  # (3,3)
                detJ_g = np.linalg.det(J_g)
                if detJ_g <= 0:
                    print("WARNING: detJ <= 0 @ element", data_e[0], detJ_g)

                # Jacobian 행렬의 역행렬 계산
                J_inv_g = np.linalg.inv(J_g)

                # dN/dx(@gauss point) 계산
                dNdx_g = J_inv_g @ dNdxi_g

                # B matrix 계산
                B_g = np.zeros((6, 3*8))
                for a in range(8):
                    dNa_dx_g = dNdx_g[0, a]
                    dNa_dy_g = dNdx_g[1, a]
                    dNa_dz_g = dNdx_g[2, a]
                    B_g[0, 3*a + 0] = dNa_dx_g
                    B_g[1, 3*a + 1] = dNa_dy_g
                    B_g[2, 3*a + 2] = dNa_dz_g
                    B_g[3, 3*a + 1] = dNa_dz_g; B_g[3, 3*a + 2] = dNa_dy_g
                    B_g[4, 3*a + 0] = dNa_dz_g; B_g[4, 3*a + 2] = dNa_dx_g
                    B_g[5, 3*a + 0] = dNa_dy_g; B_g[5, 3*a + 1] = dNa_dx_g

                # Ke(@gauss point) 계산
                Ke += w * B_g.transpose() @ C @ B_g * np.linalg.det(J_g)
    return Ke

# Element material properties matrix 계산 함수 : Tet4 요소 전용
def e_mat_prop_tet4(data_e:np.ndarray, ndarray_nodes:np.ndarray, C:np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    data_e : (6,) int ndarray
        elem_tag, elem_type, element connectivity (4 node indices)
    ndarray_nodes : (nnode, 4) float ndarray
        node_tag, Global nodal coordinates(x,y,z)
    C : (6, 6) float ndarray
        Material property matrix

    Returns
    -------
    Ke : (3*4, 3*4) float ndarray
        Element material properties matrix for Tet4
    """

    # Gauss 점과 가중치 (Tet4 선형 요소 → 1점 적분으로 정확)
    # volume (barycentric) coordinates:
    #   L1 = L2 = L3 = L4 = 1/4 (centroid)
    L2_g_points = [0.25]
    L3_g_points = [0.25]
    L4_g_points = [0.25]
    w_g  = [1.0]

    # Element별 보유 node coordinates 추출 (Tet4: 4개 노드)
    node_coords = []
    for tag_node in data_e[2:6]:
        node_coords.append(ndarray_nodes[tag_node - 1, 1:4])
    node_coords = np.array(node_coords)  # (4,3)

    # Element material properties matrix 초기화
    Ke = np.zeros((3*4, 3*4))

    # Gauss 점 하나씩 순회 (실제로는 1점)
    for i_g, L2_g in enumerate(L2_g_points):
        for j_g, L3_g in enumerate(L3_g_points):
            for k_g, L4_g in enumerate(L4_g_points):

                # L1은 종속 좌표
                L1 = 1.0 - L2_g - L3_g - L4_g
                w  = w_g[i_g] * w_g[j_g] * w_g[k_g]

                # ------------------------------------------------------------
                # Tet4 shape functions (volume / barycentric coordinates)
                #
                # N1 = L1
                # N2 = L2
                # N3 = L3
                # N4 = L4
                #
                # reference coordinate gradients (상수벡터):
                # dN/dL
                # ------------------------------------------------------------
                dNdL_g = np.array([
                    [-1.0,  1.0, 0.0, 0.0],  # dN/dL2
                    [-1.0,  0.0, 1.0, 0.0],  # dN/dL3
                    [-1.0,  0.0, 0.0, 1.0],  # dN/dL4
                ])  # (3,4)

                # Jacobian 행렬 계산 (volume coordinates → physical space)
                x1, y1, z1 = node_coords[0]
                x2, y2, z2 = node_coords[1]
                x3, y3, z3 = node_coords[2]
                x4, y4, z4 = node_coords[3]
                J_g = np.array([
                    [x2 - x1, x3 - x1, x4 - x1],
                    [y2 - y1, y3 - y1, y4 - y1],
                    [z2 - z1, z3 - z1, z4 - z1],
                ], dtype=float)
                detJ_g = np.linalg.det(J_g)
                if detJ_g <= 0:
                    print("WARNING: detJ <= 0 @ element", data_e[0], detJ_g)

                # Jacobian 역행렬
                J_inv_g = np.linalg.inv(J_g)

                # dN/dx 계산 (physical space)
                dNdx_g = J_inv_g @ dNdL_g  # (3,4)

                # B matrix 계산 (6 x 12)
                B_g = np.zeros((6, 3*4))
                for a in range(4):
                    dNa_dx = dNdx_g[0, a]
                    dNa_dy = dNdx_g[1, a]
                    dNa_dz = dNdx_g[2, a]

                    B_g[0, 3*a + 0] = dNa_dx
                    B_g[1, 3*a + 1] = dNa_dy
                    B_g[2, 3*a + 2] = dNa_dz
                    # engineering shear strains
                    B_g[3, 3*a + 1] = dNa_dz; B_g[3, 3*a + 2] = dNa_dy
                    B_g[4, 3*a + 0] = dNa_dz; B_g[4, 3*a + 2] = dNa_dx
                    B_g[5, 3*a + 0] = dNa_dy; B_g[5, 3*a + 1] = dNa_dx
                # Ke(@gauss point) 계산
                # detJ = 6 * Volume
                Ke += w * (B_g.T @ C @ B_g) * (detJ_g / 6.0)

    return Ke

# -------------------------------------------- Global material property matrix Kg 계산 --------------------------------------------
def global_mat_prop(ndarray_nodes:np.ndarray, ndarray_elements:np.ndarray, C:np.ndarray) -> lil_matrix:
    """
    Parameters
    ----------
    ndarray_nodes : (nnode, 4) float ndarray
        node_tag, Global nodal coordinates(x,y,z)
    ndarray_elements : (nelem, ?) int ndarray
        elem_tag, elem_type, element connectivity (node indices)
    C : (6, 6) float ndarray
        Material property matrix

    Returns
    -------
    Kg : (ndof, ndof) lil_matrix(sparse)
        Global material property matrix
    """
    # Global material property matrix 초기화
    ndof = ndarray_nodes.shape[0] * 3
    Kg = lil_matrix((ndof, ndof))

    # ---------------------- Element별 material property matrix 계산 및 Global material property matrix에 추가
    for data_e in ndarray_elements:
        tag_e = data_e[0]
        if data_e[1] == 5:  # hexahedral element이면
            data_e = data_e[:2+8]  # elem_tag, elem_type, element connectivity (8 node indices)
            Ke = e_mat_prop_hex8(data_e, ndarray_nodes, C)
            tag_nodes = data_e[2:10]  # hexahedral element의 node tag들
        elif data_e[1] == 4: # tetrahedral element이면
            data_e = data_e[:2+4]  # elem_tag, elem_type, element connectivity (4 node indices)
            Ke = e_mat_prop_tet4(data_e, ndarray_nodes, C)
            tag_nodes = data_e[2:6]  # tetrahedral element의 node tag들
        else: # 그 외의 element이면 패스
            continue
    
        print(f"Element {tag_e} : Adding element material property matrix to Global material property matrix")

        # Global material properties matrix에 Ke 추가
        for a_local, tag_node_a in enumerate(tag_nodes): # element의 각 node 순회(행방향)
            for b_local, tag_node_b in enumerate(tag_nodes): # element의 각 node 순회(열방향)
                for i in range(3): # 각 node의 dof 순회(행방향)
                    for j in range(3): # 각 node의 dof 순회(열방향)
                        Kg[(tag_node_a - 1)*3 + i, (tag_node_b - 1)*3 + j] += Ke[a_local*3 + i, b_local*3 + j]
    
    return Kg